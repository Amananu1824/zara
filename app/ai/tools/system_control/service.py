"""Local operating-system handlers for Zara System Control A1."""

from __future__ import annotations

import ctypes
import ipaddress
import logging
import os
import platform
import re
import socket
import subprocess
import time

from datetime import datetime
from pathlib import Path
from typing import Any

from app.ai.tools.contracts import ToolResult
from app.ai.tools.execution import (
    ToolCancelledError,
    ToolDeadlineExceeded,
)
from app.ai.tools.system_control.definitions import (
    SYSTEM_CONTROL_TOOL_NAMES,
)
from app.ai.tools.system_control.intent import (
    detect_system_control_intent,
)


logger = logging.getLogger(__name__)

try:
    import psutil
except ImportError:
    psutil = None


class SystemControlError(RuntimeError):
    """Safe, user-facing system-control error."""

    def __init__(
        self,
        code: str,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class SystemControlService:
    """Trusted handlers registered through UnifiedActionRouter."""

    DEFAULT_VOLUME_STEP = 5
    DEFAULT_BRIGHTNESS_STEP = 5

    DEFAULT_WOL_BROADCAST = "255.255.255.255"
    WOL_PORT = 9
    WOL_PACKET_COUNT = 3

    def register_handlers(self, router: Any) -> None:
        """Register all 12 tools and their independent verifiers."""

        router.register_request_guard(
            "system_control",
            self._request_guard,
        )

        confirmation_previews = {
            "power_off": self._preview_power_off,
            "restart_pc": self._preview_restart,
            "lock_pc": self._preview_lock,
            "wake_on_lan": self._preview_wake_on_lan,
        }

        result_verifiers = {
            "volume_control": self._verify_volume_control,
            "brightness_control": self._verify_brightness_control,
            "power_off": self._verify_power_action,
            "restart_pc": self._verify_power_action,
            "lock_pc": self._verify_lock_action,
            "wake_on_lan": self._verify_wake_on_lan,
        }

        for tool_name in SYSTEM_CONTROL_TOOL_NAMES:
            router.register_handler(
                tool_name,
                self._make_handler(tool_name),
                confirmation_preview=confirmation_previews.get(
                    tool_name
                ),
                result_verifier=result_verifiers.get(
                    tool_name
                ),
            )

    @staticmethod
    def _request_guard(
        request: Any,
        arguments: dict[str, Any],
    ) -> bool:
        """
        Validate the complete original user message again at the router.

        Tool arguments or model output cannot create user permission.
        """

        return (
            detect_system_control_intent(
                getattr(request, "user_message", "")
            )
            == getattr(request, "tool_name", None)
        )

    def _make_handler(
        self,
        tool_name: str,
    ):
        """Create the common execution wrapper for one handler."""

        def handler(
            arguments: dict[str, Any],
            context: Any,
        ) -> dict[str, Any]:

            try:
                context.checkpoint()

                method = getattr(
                    self,
                    f"_{tool_name}",
                )

                return method(arguments)

            except (
                ToolCancelledError,
                ToolDeadlineExceeded,
            ):
                # Let the shared execution runtime classify cancellation
                # and deadline outcomes.
                raise

            except SystemControlError as exc:
                return self._result(
                    tool_name,
                    "failed",
                    exc.message,
                    error_code=exc.code,
                )

            except PermissionError:
                return self._result(
                    tool_name,
                    "denied",
                    (
                        "Windows ne is operation ke liye permission "
                        "nahi di. Zara ne operation ko force nahi kiya."
                    ),
                    error_code="system_permission_denied",
                )

            except subprocess.TimeoutExpired:
                return self._result(
                    tool_name,
                    "failed",
                    (
                        "Windows system command time limit ke andar "
                        "complete nahi hua. Result ko verified success "
                        "nahi maana gaya."
                    ),
                    error_code="system_command_timeout",
                )

            except OSError as exc:
                logger.warning(
                    "System tool %s failed with OS error %s.",
                    tool_name,
                    type(exc).__name__,
                )

                return self._result(
                    tool_name,
                    "failed",
                    (
                        "Operating system ne requested operation "
                        "complete nahi kiya."
                    ),
                    error_code="system_os_error",
                )

            except Exception as exc:
                logger.warning(
                    "System tool %s failed with %s.",
                    tool_name,
                    type(exc).__name__,
                )

                return self._result(
                    tool_name,
                    "failed",
                    (
                        "System operation complete ya verify nahi "
                        "ho paya. Zara success claim nahi karegi."
                    ),
                    error_code="system_control_failed",
                )

        return handler

    @staticmethod
    def _result(
        tool_name: str,
        status: str,
        message: str,
        *,
        data: dict[str, Any] | None = None,
        error_code: str | None = None,
    ) -> dict[str, Any]:
        return ToolResult(
            tool_name=tool_name,
            status=status,
            message=message,
            data=data or {},
            error_code=error_code,
        ).to_legacy_dict()

    @staticmethod
    def _require_psutil():
        if psutil is None:
            raise SystemControlError(
                "dependency_missing",
                (
                    "System monitoring library missing hai. "
                    "Project dependencies install karni hongi."
                ),
            )

        return psutil

    @staticmethod
    def _require_windows() -> None:
        if os.name != "nt":
            raise SystemControlError(
                "unsupported_platform",
                "Yeh operation is build mein Windows-only hai.",
            )

    # ============================================================
    # 1. system_info
    # ============================================================

    def _system_info(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:

        system = self._require_psutil()
        uname = platform.uname()

        memory = system.virtual_memory()
        cpu_frequency = system.cpu_freq()

        processor = (
            platform.processor()
            or uname.processor
            or "Unknown"
        )

        return self._result(
            "system_info",
            "success",
            "Tumhare PC ki current system details mil gayi hain.",
            data={
                "operating_system": uname.system,
                "os_release": uname.release,
                "os_version": uname.version,
                "architecture": platform.architecture()[0],
                "machine_type": uname.machine,
                "processor": processor,
                "physical_cpu_cores": system.cpu_count(
                    logical=False
                ),
                "logical_cpu_cores": system.cpu_count(
                    logical=True
                ),
                "cpu_frequency_mhz": (
                    round(cpu_frequency.current, 2)
                    if cpu_frequency is not None
                    else None
                ),
                "ram_total_gb": round(
                    memory.total / (1024 ** 3),
                    2,
                ),
                "ram_available_gb": round(
                    memory.available / (1024 ** 3),
                    2,
                ),
                "ram_used_percent": round(
                    memory.percent,
                    1,
                ),
                "python_version": platform.python_version(),
            },
        )

    # ============================================================
    # 2. get_current_time
    # ============================================================

    def _get_current_time(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:

        now = datetime.now().astimezone()

        return self._result(
            "get_current_time",
            "success",
            f"PC ka local time {now.strftime('%I:%M:%S %p')} hai.",
            data={
                "local_time": now.isoformat(
                    timespec="seconds"
                ),
                "time_24_hour": now.strftime("%H:%M:%S"),
                "time_12_hour": now.strftime("%I:%M:%S %p"),
                "timezone_name": now.tzname(),
                "utc_offset": str(now.utcoffset()),
            },
        )

    # ============================================================
    # 3. get_current_date
    # ============================================================

    def _get_current_date(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:

        now = datetime.now().astimezone()

        return self._result(
            "get_current_date",
            "success",
            f"PC ki local date {now.strftime('%d %B %Y')} hai.",
            data={
                "local_date": now.date().isoformat(),
                "formatted_date": now.strftime("%d %B %Y"),
                "weekday": now.strftime("%A"),
                "timezone_name": now.tzname(),
            },
        )

    # ============================================================
    # 4. battery_status
    # ============================================================

    def _battery_status(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:

        system = self._require_psutil()
        battery = system.sensors_battery()

        if battery is None:
            return self._result(
                "battery_status",
                "not_found",
                "Is device par battery hardware detect nahi hui.",
                data={
                    "available": False,
                },
                error_code="battery_not_available",
            )

        seconds_left = battery.secsleft

        unknown_values = {
            getattr(system, "POWER_TIME_UNKNOWN", -1),
            getattr(system, "POWER_TIME_UNLIMITED", -2),
        }

        remaining_seconds = (
            int(seconds_left)
            if (
                isinstance(seconds_left, (int, float))
                and seconds_left not in unknown_values
                and seconds_left >= 0
            )
            else None
        )

        percent = round(float(battery.percent), 1)

        charging = (
            bool(battery.power_plugged)
            if battery.power_plugged is not None
            else None
        )

        if charging is True:
            charging_text = "Charging / plugged in"
        elif charging is False:
            charging_text = "On battery"
        else:
            charging_text = "Charging state unavailable"

        return self._result(
            "battery_status",
            "success",
            (
                f"Battery {percent}% hai. "
                f"Status: {charging_text}."
            ),
            data={
                "available": True,
                "percentage": percent,
                "plugged_in": charging,
                "charging_status": charging_text,
                "remaining_seconds": remaining_seconds,
                "remaining_minutes": (
                    round(remaining_seconds / 60, 1)
                    if remaining_seconds is not None
                    else None
                ),
            },
        )

    # ============================================================
    # 5. storage_monitor
    # ============================================================

    def _storage_monitor(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:

        system = self._require_psutil()

        partitions = system.disk_partitions(all=False)

        drives: list[dict[str, Any]] = []

        for partition in partitions:
            mountpoint = str(
                partition.mountpoint or ""
            )

            device = str(
                partition.device or ""
            )

            options = str(
                partition.opts or ""
            ).casefold()

            # Exclude optical drives and network-mounted drives.
            if (
                not mountpoint
                or "cdrom" in options
                or device.startswith("\\\\")
            ):
                continue

            try:
                usage = system.disk_usage(mountpoint)

            except (
                system.AccessDenied,
                OSError,
            ):
                continue

            total = int(usage.total)
            used = int(usage.used)
            free = int(usage.free)

            if total <= 0:
                continue

            drives.append({
                "device": device[:120],
                "mountpoint": mountpoint[:120],
                "filesystem": str(partition.fstype or "")[:40],
                "total_gb": round(total / (1024 ** 3), 2),
                "used_gb": round(used / (1024 ** 3), 2),
                "free_gb": round(free / (1024 ** 3), 2),
                "used_percent": round(
                    (used / total) * 100,
                    1,
                ),
            })

        if not drives:
            return self._result(
                "storage_monitor",
                "not_found",
                "Koi readable local storage volume detect nahi hua.",
                data={
                    "drives": [],
                },
                error_code="no_storage_volumes",
            )

        return self._result(
            "storage_monitor",
            "success",
            f"{len(drives)} local storage volume(s) ki details mil gayi hain.",
            data={
                "drive_count": len(drives),
                "drives": drives,
            },
        )

    # ============================================================
    # 6. process_monitor
    # ============================================================

    def _process_monitor(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:

        system = self._require_psutil()

        limit = arguments.get("limit", 15)

        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 50
        ):
            raise SystemControlError(
                "invalid_process_limit",
                "Process monitor limit 1 se 50 ke beech honi chahiye.",
            )

        processes: list[dict[str, Any]] = []

        for process in system.process_iter(
            attrs=[
                "pid",
                "name",
                "status",
                "memory_info",
            ]
        ):
            try:
                info = process.info

                memory_info = info.get("memory_info")
                rss = (
                    int(memory_info.rss)
                    if memory_info is not None
                    else 0
                )

                processes.append({
                    "pid": int(info.get("pid") or 0),
                    "name": str(
                        info.get("name") or "Unknown"
                    )[:100],
                    "status": str(
                        info.get("status") or "unknown"
                    )[:30],
                    "memory_mb": round(
                        rss / (1024 ** 2),
                        1,
                    ),
                })

            except (
                system.NoSuchProcess,
                system.AccessDenied,
                system.ZombieProcess,
            ):
                continue

        processes.sort(
            key=lambda item: item["memory_mb"],
            reverse=True,
        )

        selected = processes[:limit]

        return self._result(
            "process_monitor",
            "success",
            (
                f"{len(selected)} running process entries mil gayi hain. "
                "Kisi process ko stop ya modify nahi kiya gaya."
            ),
            data={
                "returned_count": len(selected),
                "available_count": len(processes),
                "processes": selected,
                "process_actions_performed": False,
            },
        )

    # ============================================================
    # Shared audio helpers
    # ============================================================

    @staticmethod
    def _audio_endpoint():
        try:
            from pycaw.pycaw import AudioUtilities

        except ImportError as exc:
            raise SystemControlError(
                "audio_dependency_missing",
                (
                    "Windows audio control library available nahi hai. "
                    "A1 dependencies install karni hongi."
                ),
            ) from exc

        try:
            speakers = AudioUtilities.GetSpeakers()
            endpoint = speakers.EndpointVolume

        except Exception as exc:
            raise SystemControlError(
                "audio_device_unavailable",
                (
                    "Default Windows audio output device access nahi "
                    "ho paya."
                ),
            ) from exc

        if endpoint is None:
            raise SystemControlError(
                "audio_endpoint_unavailable",
                "Default audio endpoint available nahi hai.",
            )

        return endpoint

    @classmethod
    def _read_volume_state(
        cls,
    ) -> dict[str, Any]:

        endpoint = cls._audio_endpoint()

        scalar = float(
            endpoint.GetMasterVolumeLevelScalar()
        )

        return {
            "volume_percent": round(
                max(0.0, min(1.0, scalar)) * 100
            ),
            "muted": bool(
                endpoint.GetMute()
            ),
        }

    # ============================================================
    # 7. volume_control
    # ============================================================

    def _volume_control(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:

        endpoint = self._audio_endpoint()

        action = arguments.get("action")

        current_scalar = float(
            endpoint.GetMasterVolumeLevelScalar()
        )

        current = round(
            max(0.0, min(1.0, current_scalar)) * 100
        )

        current_muted = bool(
            endpoint.GetMute()
        )

        if action == "mute":
            if current_muted:
                return self._result(
                    "volume_control",
                    "unchanged",
                    "Audio pehle se muted hai.",
                    data={
                        "action": action,
                        "volume_percent": current,
                        "muted": True,
                    },
                )

            endpoint.SetMute(1, None)

        elif action == "unmute":
            if not current_muted:
                return self._result(
                    "volume_control",
                    "unchanged",
                    "Audio pehle se unmuted hai.",
                    data={
                        "action": action,
                        "volume_percent": current,
                        "muted": False,
                    },
                )

            endpoint.SetMute(0, None)

        elif action == "set":
            target = arguments.get("level")

            if target is None:
                raise SystemControlError(
                    "volume_level_required",
                    "Volume set karne ke liye level 0-100 dena hoga.",
                )

            target = int(target)

            if target == current:
                return self._result(
                    "volume_control",
                    "unchanged",
                    f"Volume pehle se {current}% hai.",
                    data={
                        "action": action,
                        "volume_percent": current,
                        "muted": current_muted,
                    },
                )

            endpoint.SetMasterVolumeLevelScalar(
                target / 100.0,
                None,
            )

        elif action in {"increase", "decrease"}:
            step = int(
                arguments.get(
                    "step",
                    self.DEFAULT_VOLUME_STEP,
                )
            )

            if not 1 <= step <= 25:
                raise SystemControlError(
                    "invalid_volume_step",
                    "Volume step 1-25 ke beech honi chahiye.",
                )

            target = (
                current + step
                if action == "increase"
                else current - step
            )

            target = max(0, min(100, target))

            if target == current:
                return self._result(
                    "volume_control",
                    "unchanged",
                    f"Volume already {current}% par hai.",
                    data={
                        "action": action,
                        "volume_percent": current,
                        "muted": current_muted,
                    },
                )

            endpoint.SetMasterVolumeLevelScalar(
                target / 100.0,
                None,
            )

        else:
            raise SystemControlError(
                "invalid_volume_action",
                "Unsupported volume action.",
            )

        # Read back the real state instead of trusting the setter call.
        observed = self._read_volume_state()

        if action == "mute":
            if observed["muted"] is not True:
                raise SystemControlError(
                    "volume_mute_unverified",
                    "Windows audio mute state verify nahi ho paya.",
                )

        elif action == "unmute":
            if observed["muted"] is not False:
                raise SystemControlError(
                    "volume_unmute_unverified",
                    "Windows audio unmute state verify nahi ho paya.",
                )

        else:
            if abs(
                observed["volume_percent"] - target
            ) > 2:
                raise SystemControlError(
                    "volume_change_unverified",
                    "Requested volume actual device state se match nahi hua.",
                )

        return self._result(
            "volume_control",
            "success",
            (
                f"Volume control action complete hai. "
                f"Current volume {observed['volume_percent']}% hai."
            ),
            data={
                "action": action,
                "volume_percent": observed["volume_percent"],
                "muted": observed["muted"],
            },
        )

    def _verify_volume_control(
        self,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> bool:

        observed = self._read_volume_state()
        data = result.get("data") or {}

        action = arguments.get("action")

        if action == "mute":
            return observed["muted"] is True

        if action == "unmute":
            return observed["muted"] is False

        expected = data.get("volume_percent")

        if not isinstance(expected, (int, float)):
            return False

        return (
            abs(observed["volume_percent"] - expected) <= 2
        )

    # ============================================================
    # Shared brightness helpers
    # ============================================================

    @staticmethod
    def _brightness_library():
        try:
            import screen_brightness_control as sbc

        except ImportError as exc:
            raise SystemControlError(
                "brightness_dependency_missing",
                (
                    "Display brightness library available nahi hai. "
                    "A1 dependencies install karni hongi."
                ),
            ) from exc

        return sbc

    @classmethod
    def _read_brightness(
        cls,
        display: int,
    ) -> int:

        sbc = cls._brightness_library()

        raw = sbc.get_brightness(
            display=display
        )

        if isinstance(raw, list):
            if not raw:
                raise SystemControlError(
                    "display_not_found",
                    "Requested display brightness available nahi hai.",
                )

            raw = raw[0]

        try:
            value = float(raw)

        except (TypeError, ValueError) as exc:
            raise SystemControlError(
                "invalid_brightness_result",
                "Display ne valid brightness value return nahi ki.",
            ) from exc

        return max(
            0,
            min(100, round(value)),
        )

    # ============================================================
    # 8. brightness_control
    # ============================================================

    def _brightness_control(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:

        sbc = self._brightness_library()

        action = arguments.get("action")
        display = int(arguments.get("display", 0))

        if not 0 <= display <= 8:
            raise SystemControlError(
                "invalid_display",
                "Display index 0-8 ke beech honi chahiye.",
            )

        current = self._read_brightness(display)

        if action == "set":
            target = arguments.get("level")

            if target is None:
                raise SystemControlError(
                    "brightness_level_required",
                    "Brightness set karne ke liye level 0-100 dena hoga.",
                )

            target = int(target)

        elif action in {"increase", "decrease"}:
            step = int(
                arguments.get(
                    "step",
                    self.DEFAULT_BRIGHTNESS_STEP,
                )
            )

            if not 1 <= step <= 25:
                raise SystemControlError(
                    "invalid_brightness_step",
                    "Brightness step 1-25 ke beech honi chahiye.",
                )

            target = (
                current + step
                if action == "increase"
                else current - step
            )

        else:
            raise SystemControlError(
                "invalid_brightness_action",
                "Unsupported brightness action.",
            )

        target = max(0, min(100, target))

        if target == current:
            return self._result(
                "brightness_control",
                "unchanged",
                (
                    f"Display {display} ki brightness pehle se "
                    f"{current}% hai."
                ),
                data={
                    "action": action,
                    "display_index": display,
                    "brightness_percent": current,
                },
            )

        sbc.set_brightness(
            target,
            display=display,
        )

        observed = self._read_brightness(display)

        if abs(observed - target) > 2:
            raise SystemControlError(
                "brightness_change_unverified",
                (
                    "Requested brightness actual display value se "
                    "match nahi hui."
                ),
            )

        return self._result(
            "brightness_control",
            "success",
            (
                f"Display {display} ki brightness "
                f"{observed}% set ho gayi."
            ),
            data={
                "action": action,
                "display_index": display,
                "brightness_percent": observed,
            },
        )

    def _verify_brightness_control(
        self,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> bool:

        data = result.get("data") or {}

        display = data.get("display_index")
        expected = data.get("brightness_percent")

        if (
            not isinstance(display, int)
            or not isinstance(expected, (int, float))
        ):
            return False

        observed = self._read_brightness(display)

        return abs(observed - expected) <= 2

    # ============================================================
    # Windows lock and power helpers
    # ============================================================

    @staticmethod
    def _shutdown_exe() -> Path:
        root = os.environ.get(
            "SystemRoot",
            r"C:\Windows",
        )

        executable = (
            Path(root)
            / "System32"
            / "shutdown.exe"
        )

        if not executable.is_file():
            raise SystemControlError(
                "shutdown_executable_missing",
                "Windows shutdown executable locate nahi hua.",
            )

        return executable

    def _request_power_action(
        self,
        *,
        restart: bool,
    ) -> dict[str, Any]:

        self._require_windows()

        executable = self._shutdown_exe()

        flag = "/r" if restart else "/s"

        # Fixed executable and fixed arguments. No shell, no user command,
        # no /f forced application closure.
        command = [
            str(executable),
            flag,
            "/t",
            "0",
            "/c",
            "Zara approved local system action",
        ]

        completed = subprocess.run(
            command,
            shell=False,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            creationflags=getattr(
                subprocess,
                "CREATE_NO_WINDOW",
                0,
            ),
        )

        if completed.returncode != 0:
            raise SystemControlError(
                "windows_power_request_rejected",
                (
                    "Windows ne power request accept nahi ki. "
                    "Zara ise successful nahi kahegi."
                ),
            )

        action = "restart" if restart else "shutdown"

        return self._result(
            "restart_pc" if restart else "power_off",
            "success",
            (
                f"Windows ne {action} request accept kar li hai. "
                "Actual shutdown/restart Windows handle karega. "
                "Kisi application ko force-close karne ka flag use nahi hua."
            ),
            data={
                "requested_action": action,
                "accepted_by_windows": True,
                "return_code": int(completed.returncode),
                "forced_app_close": False,
                "actual_power_transition_verified": False,
            },
        )

    # ============================================================
    # 9. power_off
    # ============================================================

    def _power_off(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:

        return self._request_power_action(
            restart=False
        )

    # ============================================================
    # 10. restart_pc
    # ============================================================

    def _restart_pc(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:

        return self._request_power_action(
            restart=True
        )

    @staticmethod
    def _verify_power_action(
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> bool:

        data = result.get("data") or {}

        # This verifies Windows accepted the request.
        # It does not claim the physical power transition completed.
        return (
            data.get("accepted_by_windows") is True
            and data.get("return_code") == 0
            and data.get("forced_app_close") is False
        )

    # ============================================================
    # 11. lock_pc
    # ============================================================

    def _lock_pc(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:

        self._require_windows()

        user32 = ctypes.WinDLL(
            "user32",
            use_last_error=True,
        )

        lock_workstation = user32.LockWorkStation
        lock_workstation.argtypes = []
        lock_workstation.restype = ctypes.c_bool

        accepted = bool(
            lock_workstation()
        )

        if not accepted:
            error_code = ctypes.get_last_error()

            raise SystemControlError(
                "windows_lock_request_rejected",
                (
                    "Windows ne lock request accept nahi ki. "
                    f"Windows error code: {error_code}."
                ),
            )

        return self._result(
            "lock_pc",
            "success",
            (
                "Windows ne current session lock request accept kar li. "
                "Lock transition Windows asynchronously handle karega."
            ),
            data={
                "request_initiated": True,
                "actual_lock_state_verified": False,
            },
        )

    @staticmethod
    def _verify_lock_action(
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> bool:

        data = result.get("data") or {}

        return data.get("request_initiated") is True

    # ============================================================
    # 12. wake_on_lan
    # ============================================================

    def _wake_on_lan(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:

        raw_mac = arguments.get("mac_address")

        if not isinstance(raw_mac, str):
            raise SystemControlError(
                "invalid_mac_address",
                "Valid target MAC address dena zaroori hai.",
            )

        compact_mac = re.sub(
            r"[:-]",
            "",
            raw_mac.strip(),
        ).casefold()

        if not re.fullmatch(
            r"[0-9a-f]{12}",
            compact_mac,
        ):
            raise SystemControlError(
                "invalid_mac_address",
                "MAC address ka format valid nahi hai.",
            )

        mac_bytes = bytes.fromhex(compact_mac)

        if (
            mac_bytes == b"\x00" * 6
            or mac_bytes == b"\xff" * 6
            or (mac_bytes[0] & 1) != 0
        ):
            raise SystemControlError(
                "invalid_unicast_mac",
                "Wake-on-LAN ke liye valid unicast MAC address chahiye.",
            )

        raw_broadcast = arguments.get(
            "broadcast_address",
            self.DEFAULT_WOL_BROADCAST,
        )

        try:
            broadcast = ipaddress.IPv4Address(
                raw_broadcast
            )

        except ipaddress.AddressValueError as exc:
            raise SystemControlError(
                "invalid_broadcast_address",
                "Broadcast address valid IPv4 address nahi hai.",
            ) from exc

        limited_broadcast = ipaddress.IPv4Address(
            self.DEFAULT_WOL_BROADCAST
        )

        if (
            broadcast.is_multicast
            or broadcast.is_loopback
            or broadcast.is_unspecified
            or (
                broadcast.is_reserved
                and broadcast != limited_broadcast
            )
            or (
                not broadcast.is_private
                and broadcast != limited_broadcast
            )
        ):
            raise SystemControlError(
                "broadcast_address_blocked",
                (
                    "Wake-on-LAN broadcast sirf limited broadcast "
                    "ya private local-network address par allowed hai."
                ),
            )

        packet = (
            b"\xff" * 6
            + mac_bytes * 16
        )

        packets_sent = 0

        with socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        ) as sock:

            sock.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_BROADCAST,
                1,
            )

            for _ in range(self.WOL_PACKET_COUNT):
                sent = sock.sendto(
                    packet,
                    (
                        str(broadcast),
                        self.WOL_PORT,
                    ),
                )

                if sent != len(packet):
                    raise SystemControlError(
                        "wake_packet_incomplete",
                        "Wake-on-LAN packet complete send nahi hua.",
                    )

                packets_sent += 1

                time.sleep(0.05)

        return self._result(
            "wake_on_lan",
            "success",
            (
                "Wake-on-LAN magic packets local network par send ho gaye. "
                "Target PC wake hoga ya nahi, yeh uske BIOS, network adapter "
                "aur network configuration par depend karta hai."
            ),
            data={
                "target_mac": ":".join(
                    compact_mac[index:index + 2]
                    for index in range(0, 12, 2)
                ),
                "broadcast_address": str(broadcast),
                "udp_port": self.WOL_PORT,
                "datagrams_sent": packets_sent,
                "local_send_accepted": (
                    packets_sent == self.WOL_PACKET_COUNT
                ),
                "target_wake_verified": False,
            },
        )

    @staticmethod
    def _verify_wake_on_lan(
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> bool:

        data = result.get("data") or {}

        return (
            data.get("local_send_accepted") is True
            and data.get("datagrams_sent")
            == SystemControlService.WOL_PACKET_COUNT
            and data.get("target_wake_verified") is False
        )

    # ============================================================
    # Confirmation previews
    # ============================================================

    @staticmethod
    def _preview_power_off(
        arguments: dict[str, Any],
    ) -> str:
        return (
            "SHUT DOWN this local Windows PC now. "
            "Windows will not be instructed to force-close applications. "
            "Save important work before approving."
        )

    @staticmethod
    def _preview_restart(
        arguments: dict[str, Any],
    ) -> str:
        return (
            "RESTART this local Windows PC now. "
            "Windows will not be instructed to force-close applications. "
            "Save important work before approving."
        )

    @staticmethod
    def _preview_lock(
        arguments: dict[str, Any],
    ) -> str:
        return (
            "LOCK the current Windows interactive session. "
            "Your normal Windows sign-in will be needed to unlock it."
        )

    @staticmethod
    def _preview_wake_on_lan(
        arguments: dict[str, Any],
    ) -> str:

        mac = arguments["mac_address"]

        broadcast = arguments.get(
            "broadcast_address",
            SystemControlService.DEFAULT_WOL_BROADCAST,
        )

        return (
            f"Send Wake-on-LAN packets to MAC {mac} "
            f"using broadcast address {broadcast}. "
            "This sends network packets; it does not guarantee "
            "that the target computer wakes."
        )