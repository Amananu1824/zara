"""Central conversation intent, emotion and response-move selection."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal


ResponseMode = Literal[
    "listen",
    "repair",
    "answer",
    "acknowledge",
    "support",
    "converse",
    "continue",
]

EmotionSignal = Literal[
    "positive",
    "frustrated",
    "sad",
    "worried",
    "playful",
    "neutral_or_unclear",
]

ResponseTone = Literal["warm", "calm", "gentle", "playful", "natural"]
EmotionConfidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class ConversationDecision:
    intent: str
    response_mode: ResponseMode
    emotion: EmotionSignal
    tone: ResponseTone
    prefer_one_sentence: bool
    uses_recent_context: bool
    avoid_unsolicited_advice: bool
    avoid_routine_help_offers: bool
    has_explicit_request: bool
    emotion_confidence: EmotionConfidence = "low"
    invites_story_question: bool = False

    @property
    def prompt_instruction(self) -> str:
        """Translate the selected decision into compact response guidance."""
        if self.intent == "greeting":
            instruction = (
                "GREETING: Return a natural, brief greeting. Do not invent "
                "a plan, activity, event, or topic. Do not append an unrelated "
                "question or a routine offer of help."
            )

        elif self.intent == "social_check_in":
            instruction = (
                "SOCIAL CHECK-IN: The user is asking how Zara is doing. "
                "Answer that question directly and naturally first. Do not "
                "reverse the question without answering, invent a plan or "
                "human experience, change the subject, or add an unrelated "
                "help offer. Keep the reply brief and warm."
            )

        elif self.intent == "mutual_check_in":
            instruction = (
                "MUTUAL CHECK-IN: Briefly acknowledge the user's stated "
                "condition and answer their question about Zara. Keep the "
                "wording natural and respectful. Do not invent plans, "
                "experiences, or unrelated topics."
            )

        elif self.intent == "style_boundary":
            instruction = (
                "STYLE BOUNDARY: The user has corrected Zara's behaviour "
                "or rejected praise. Acknowledge the specific boundary once "
                "if needed, then follow it. Do not praise again, defend the "
                "previous response, or append a generic help offer."
            )

        elif self.intent == "correction":
            instruction = (
                "REPAIR: Address the specific mistake or complaint. "
                "Acknowledge it briefly when established, correct it using "
                "available evidence, and do not defend or repeat the bad "
                "answer. Do not invent facts to make an apology convincing."
            )
            if self.has_explicit_request:
                instruction += (
                    " Fulfil the user's explicit request after addressing "
                    "the correction; do not stop at an apology."
                )

        elif self.intent == "relationship_question":
            instruction = (
                "RELATIONSHIP: Answer directly and warmly within Zara's "
                "configured platonic AI-friend persona. Do not imply "
                "romantic attachment, jealousy, possessiveness, exclusivity, "
                "or pressure the user to depend on Zara."
            )

        elif self.intent == "listening_request":
            instruction = (
                "LISTEN: The user wants to vent or has declined advice. "
                "Give a brief, proportionate acknowledgement. Do not offer "
                "solutions, coping steps, troubleshooting, or a follow-up "
                "question. Respect the request to listen."
            )

        elif self.intent == "emotional_sharing":
            instruction = (
                "EMOTIONAL SHARING: Respond gently to what the user "
                "actually disclosed. Do not guess what happened, invent "
                "circumstances, impose a positive interpretation, or turn "
                "the disclosure into troubleshooting."
            )
            if self.invites_story_question:
                instruction += (
                    " The user appears to be opening a story about a bad "
                    "experience. Acknowledge that briefly and ask one "
                    "natural, directly relevant question about what happened."
                )
            else:
                instruction += (
                    " Acknowledge the feeling without forcing a question "
                    "or pretending to know its cause."
                )

        elif self.intent == "positive_sharing":
            instruction = (
                "POSITIVE SHARING: Recognize the actual progress or good "
                "news in a natural, proportionate way. Do not invent details, "
                "overpraise the user, offer unrequested advice, or ask what "
                "they will do next."
            )

        elif self.intent == "personal_sharing":
            instruction = (
                "PERSONAL UPDATE: React to the detail the user actually "
                "shared. Do not rate their work, assume success or failure, "
                "repeat the message mechanically, or turn the update into "
                "a task or routine follow-up question."
            )

        elif self.intent == "explicit_request":
            instruction = (
                "DIRECT REQUEST: Fulfil the actual request first. Use the "
                "relevant history and evidence. If an emotion is clearly "
                "present, reflect it briefly without derailing the answer."
            )

        elif self.intent == "contextual_follow_up":
            instruction = (
                "CONTEXTUAL FOLLOW-UP: Resolve short references using the "
                "relevant recent conversation. Do not restart the topic or "
                "ask for information already provided. Answer the follow-up "
                "rather than giving a generic response."
            )

        elif self.intent == "social_conversation":
            instruction = (
                "CASUAL CONVERSATION: Respond to the actual social meaning. "
                "A light playful tone is appropriate only when the message "
                "supports it. Do not force jokes, catchphrases, or questions."
            )

        else:
            instruction = (
                "GENERAL CONVERSATION: Respond naturally to the message "
                "itself. Do not invent a mood, event, plan, or backstory. "
                "Do not add a follow-up question merely to keep the "
                "conversation going."
            )

        instructions = [
            instruction,
            f"TONE: Use a {self.tone} tone where it fits. Express it in the "
            "wording; do not announce or explain the selected emotion.",
        ]

        if self.emotion_confidence == "low":
            instructions.append(
                "EMOTION UNCERTAINTY: No clear emotion is established. "
                "Do not assume one or describe the user as feeling something."
            )
        elif self.emotion_confidence == "medium":
            instructions.append(
                "EMOTION UNCERTAINTY: Treat the inferred emotion as tentative. "
                "Do not exaggerate it or assume its cause."
            )

        if self.avoid_unsolicited_advice:
            instructions.append(
                "Avoid unsolicited advice and solutions. Give practical "
                "guidance when the user asks for it or when it is necessary "
                "to fulfil their direct request."
            )

        if self.avoid_routine_help_offers:
            instructions.append(
                "Do not habitually ask whether the user needs help or append "
                "a generic closing question. Ask only a question that serves "
                "the current conversation or is essential to the request."
            )

        if self.uses_recent_context:
            instructions.append(
                "Use relevant recent context to resolve references, but do "
                "not introduce unrelated details from earlier turns."
            )

        return "\n".join(instructions)


# A complete greeting, not a greeting prefix attached to an unrelated task.
_GREETING_PATTERN = re.compile(
    r"^\s*(?:hi+|hello+|hey+|salaam|salam|namaste|"
    r"good\s+(?:morning|afternoon|evening|night))"
    r"(?:\s+zara)?[\s!.,🙂😊👋✨😂🤣]*$",
    re.IGNORECASE,
)

_ASKS_ZARA_STATUS_PATTERN = re.compile(
    r"\b(?:how\s+are\s+you(?:\s+doing)?|how'?s\s+it\s+going|"
    r"what'?s\s+up|kya\s+haal(?:\s+hai)?)\b"
    r"|"
    r"\b(?:zara|tum|aap|you)\b.{0,35}"
    r"\b(?:kaise|kaisi|kaisa|theek|thik|fine|okay|ok)\b"
    r"|"
    r"\b(?:kaise|kaisi|kaisa)\s+(?:ho|hain)\s+(?:aap|tum|zara)?\b",
    re.IGNORECASE,
)

_USER_STATUS_PATTERN = re.compile(
    r"\b(?:main|mai|me|i)\b.{0,30}"
    r"\b(?:thik|theek|fine|okay|ok)\b",
    re.IGNORECASE,
)

_PRAISE_BOUNDARY_PATTERN = re.compile(
    r"\b(?:meri|mujhe|mujhko)\b.{0,60}"
    r"\b(?:tareef|tarif|trif|praise|compliment)\b"
    r".{0,40}\b(?:mat|nahi|nahin|nhi)\b"
    r"|"
    r"\b(?:don't|dont|do\s+not|stop)\b.{0,40}"
    r"\b(?:praise|compliment|tareef|tarif)\b",
    re.IGNORECASE,
)

_REPEATED_HELP_OFFER_PATTERN = re.compile(
    r"\b(?:kitni\s+baar|baar\s*baar|har\s+baar|"
    r"keep\s+(?:asking|offering)|repeatedly)\b"
    r".{0,100}\b(?:help|madad|offer\w*|pooch\w*|"
    r"puch\w*|ask\w*|question\w*)\b"
    r"|"
    r"\b(?:help|madad|offer\w*|question\w*)\b"
    r".{0,80}\b(?:baar\s*baar|har\s+baar|kitni\s+baar)\b",
    re.IGNORECASE,
)

_LISTEN_ONLY_PATTERN = re.compile(
    r"\b(?:just\s+listen|only\s+listen|sirf\s+suno|bas\s+suno|"
    r"i\s+just\s+want\s+to\s+vent|just\s+let\s+me\s+vent|"
    r"no\s+solution|solution\s+nahi\s+chahiye|"
    r"solution\s+nahin\s+chahiye|mujhe\s+solution\s+(?:nahi|nahin)\s+"
    r"(?:chahiye|chaiye)|bas\s+(?:mann|man)\s+halka\s+karna\s+tha)\b",
    re.IGNORECASE,
)

_NO_ADVICE_PATTERN = re.compile(
    r"\b(?:advice|salah|suggestions?)\b"
    r".{0,60}\b(?:mat|nahi|nahin|nhi|don't|dont|do\s+not|avoid)\b"
    r"|"
    r"\b(?:don't|dont|do\s+not|avoid|mat|nahi|nahin|nhi)\b"
    r".{0,60}\b(?:advice|salah|suggestions?)\b",
    re.IGNORECASE,
)

_NO_ROUTINE_HELP_PATTERN = re.compile(
    r"\b(?:don't|dont|do\s+not|stop|never)\b.{0,50}"
    r"\b(?:keep\s+)?asking\b.{0,60}\b(?:help|need|want|whether)\b"
    r"|"
    r"\b(?:bar|baar)\s*baar\b.{0,50}\b(?:mat\s+)?"
    r"(?:pucho|poochho|puchho|pooch|puch)\b"
    r".{0,80}\b(?:help|madad|zarurat|chahiye)\b"
    r"|"
    r"\b(?:mujhe|mujhse)\b.{0,50}\b(?:bar|baar)\s*baar\b"
    r".{0,50}\b(?:pucho|poochho|puchho|pooch|puch)\b"
    r"|"
    r"\b(?:i'll|i will|main|mai)\b.{0,70}"
    r"\b(?:ask|tell|bata|bol|keh)\b.{0,50}\b(?:help|madad)\b",
    re.IGNORECASE,
)

_CORRECTION_PATTERN = re.compile(
    r"\b(?:you were wrong|that's wrong|that is wrong|"
    r"not what i asked|you repeated|stop repeating|same answer|"
    r"wrong page|you don't understand|you do not understand|"
    r"bad answer|poor answer|terrible answer|"
    r"(?:kharab|bekar|ghatiya)\s+(?:answer|jawab|response|reply)|"
    r"(?:answer|jawab|response)\s+(?:bahut\s+)?(?:kharab|bekar)|"
    r"kya\s+bakwas|galat bataya|source galat|jawab galat|"
    r"robot ki tarah|baat nahi samajh)\b",
    re.IGNORECASE,
)

_RELATIONSHIP_PATTERN = re.compile(
    r"\b(?:will you be my friend|can we be friends|"
    r"are you my friend|are we friends|"
    r"mere\s+se\s+friendship|mere\s+saath\s+friendship|"
    r"friendship\s+karogi|dosti\s+karogi|"
    r"friend\s+banogi|best\s+friend\s+banogi)\b",
    re.IGNORECASE,
)

_DIRECT_REQUEST_PATTERN = re.compile(
    r"[?؟]"
    r"|\b(?:what|why|how|where|when|who|which|kya|kyu|kyun|"
    r"kaise|kaisi|kaun|kab|kahan|tell me|show me|give me|explain|"
    r"write|generate|create|make|fix|debug|solve|compare|calculate|"
    r"summarize|summarise|translate|rewrite|implement|inspect|review|"
    r"analyze|analyse|find|search|check|continue|batao|bataiye|"
    r"samjhao|samjha do|likho|banao|kar do|de do|code do|help karo)\b",
    re.IGNORECASE,
)

_ACHIEVEMENT_PATTERN = re.compile(
    r"\b(?:chal\s+gaya|chal\s+gayi|kaam\s+kar\s+gaya|"
    r"run\s+ho\s+gaya|pass\s+ho\s+gaya|pass\s+ho\s+gayi|"
    r"complete\s+ho\s+gaya|mil\s+gaya|mil\s+gayi|"
    r"i\s+did\s+it|it\s+worked|got\s+selected|"
    r"finally\s+(?:worked|passed|completed|fixed)|works?\s+now)\b",
    re.IGNORECASE,
)

_PERSONAL_UPDATE_PATTERN = re.compile(
    r"\b(?:main|mai|me|i)\b.{0,55}"
    r"\b(?:apna|apni|apne|mera|meri|mere|my|project|code|"
    r"program|app|website|business|kaam|work|course|padhai|"
    r"learning|ai|python|coding)\b.{0,65}"
    r"\b(?:bana\w*|kar\w*|seekh\w*|padh\w*|working|building|"
    r"creating|develop\w*|trying|started|start\s+kar\w*)\b",
    re.IGNORECASE,
)

_NEGATIVE_EVENT_PATTERN = re.compile(
    r"\b(?:pata|pta)\s+hai\b.{0,100}"
    r"\b(?:mere\s+(?:saath|sath)|with\s+me)\b.{0,60}"
    r"\b(?:bura|kharab|bad|awful|rough|mushkil|difficult)\b"
    r".{0,35}\b(?:hua|hui|gaya|gayi|happened|occurred)\b"
    r"|"
    r"\b(?:mere\s+(?:saath|sath)|with\s+me)\b.{0,90}"
    r"\b(?:bura|kharab|bad|awful|rough|mushkil|difficult)\b"
    r".{0,35}\b(?:hua|hui|gaya|gayi|happened|occurred)\b"
    r"|"
    r"\b(?:aaj|today)\b.{0,80}"
    r"\b(?:bura|kharab|bad|awful|rough)\b.{0,35}"
    r"\b(?:hua|hui|gaya|gayi|happened)\b",
    re.IGNORECASE,
)

_FRUSTRATION_PATTERN = re.compile(
    r"\b(?:i(?:'m| am| feel)|main|mai|mujhe|mujhko)\b.{0,65}"
    r"\b(?:frustrat\w*|irritat\w*|angry|gussa|naraz|"
    r"fed\s+up|pareshan|tang\s+aa\s+gaya|tang\s+aa\s+gayi)\b",
    re.IGNORECASE,
)

_SAD_PATTERN = re.compile(
    r"\b(?:i(?:'m| am| feel)|main|mai|mujhe|mujhko|mera|meri)\b"
    r".{0,65}\b(?:sad|udaas|dukhi|hurt|disappointed|lonely|upset|"
    r"mood\s+off|bura\s+lag\w*)\b"
    r"|"
    r"\b(?:dil\s+dukha|mood\s+off|bura\s+lag\s+raha|"
    r"bura\s+lag\s+rahi)\b",
    re.IGNORECASE,
)

_WORRIED_PATTERN = re.compile(
    r"\b(?:i(?:'m| am| feel)|main|mai|mujhe|mujhko)\b.{0,65}"
    r"\b(?:worried|anxious|nervous|scared|concerned|chinta|tension)\b",
    re.IGNORECASE,
)

_POSITIVE_PATTERN = re.compile(
    r"\b(?:i(?:'m| am| feel)|main|mai|mujhe|mujhko)\b.{0,65}"
    r"\b(?:khush|khushi|happy|excited|proud|relieved)\b",
    re.IGNORECASE,
)

_PLAYFUL_PATTERN = re.compile(
    r"\b(?:haha+|hehe+|lmao|mazak|joke|hilarious)\b"
    r"|[😂🤣😄😆🥳🎉🤩]",
    re.IGNORECASE,
)

_NEGATION_WORD_PATTERN = re.compile(
    r"\b(?:not|never|nahi|nahin|nhi|mat)\b",
    re.IGNORECASE,
)


def _has_unnegated_match(
    pattern: re.Pattern[str],
    message: str,
) -> bool:
    """Avoid interpreting an explicitly negated emotion as present."""
    for match in pattern.finditer(message):
        matched_text = match.group(0)

        if _NEGATION_WORD_PATTERN.search(matched_text):
            continue

        prefix = message[max(0, match.start() - 18):match.start()]
        if re.search(
            r"\b(?:not|never|nahi|nahin|nhi|mat)\s*$",
            prefix,
            re.IGNORECASE,
        ):
            continue

        suffix = message[match.end():match.end() + 22]
        if re.match(
            r"\s+(?:nahi|nahin|nhi|not|never)\b",
            suffix,
            re.IGNORECASE,
        ):
            continue

        return True

    return False


def _detect_emotion(
    message: str,
) -> tuple[EmotionSignal, ResponseTone, EmotionConfidence]:
    if _has_unnegated_match(_FRUSTRATION_PATTERN, message):
        return "frustrated", "calm", "high"

    if _has_unnegated_match(_NEGATIVE_EVENT_PATTERN, message):
        return "sad", "gentle", "high"

    if _has_unnegated_match(_SAD_PATTERN, message):
        return "sad", "gentle", "high"

    if _has_unnegated_match(_WORRIED_PATTERN, message):
        return "worried", "calm", "high"

    if (
        _has_unnegated_match(_POSITIVE_PATTERN, message)
        or _has_unnegated_match(_ACHIEVEMENT_PATTERN, message)
    ):
        return "positive", "warm", "high"

    if _has_unnegated_match(_PLAYFUL_PATTERN, message):
        return "playful", "playful", "medium"

    return "neutral_or_unclear", "natural", "low"


def decide_conversation(
    user_message: str,
    recent_history: Sequence[Mapping[str, str]] | None = None,
) -> ConversationDecision:
    """Choose the response move before generation, using one lightweight pass."""
    message = re.sub(r"\s+", " ", (user_message or "").strip())
    history_available = bool(recent_history)

    greeting = bool(_GREETING_PATTERN.fullmatch(message))
    asks_zara_status = bool(_ASKS_ZARA_STATUS_PATTERN.search(message))
    user_status = bool(_USER_STATUS_PATTERN.search(message))
    social_check_in = asks_zara_status
    mutual_check_in = asks_zara_status and user_status

    praise_boundary = bool(_PRAISE_BOUNDARY_PATTERN.search(message))
    repeated_help_offer = bool(_REPEATED_HELP_OFFER_PATTERN.search(message))
    correction = bool(
        _CORRECTION_PATTERN.search(message) or repeated_help_offer
    )

    relationship_question = bool(_RELATIONSHIP_PATTERN.search(message))
    explicit_listening = bool(_LISTEN_ONLY_PATTERN.search(message))
    no_advice = bool(_NO_ADVICE_PATTERN.search(message))
    no_routine_help = bool(_NO_ROUTINE_HELP_PATTERN.search(message))
    direct_request = bool(_DIRECT_REQUEST_PATTERN.search(message))

    emotion, tone, emotion_confidence = _detect_emotion(message)
    achievement = _has_unnegated_match(_ACHIEVEMENT_PATTERN, message)
    negative_event = _has_unnegated_match(_NEGATIVE_EVENT_PATTERN, message)
    story_opening = bool(
        re.search(r"\b(?:pata\s+hai|pta\s+hai|you\s+know|guess\s+what)\b",
                  message, re.IGNORECASE)
    )

    words = re.findall(r"\b[\w']+\b", message)
    uses_recent_context = (
        history_available
        and len(words) <= 6
        and len(message) <= 48
        and not greeting
        and not social_check_in
    )

    if praise_boundary:
        intent = "style_boundary"
        mode: ResponseMode = "repair"
        tone = "calm"

    elif explicit_listening or (no_advice and not direct_request):
        intent = "listening_request"
        mode = "listen"
        tone = "gentle"

    elif correction:
        intent = "correction"
        mode = "answer" if direct_request else "repair"
        tone = "calm"

    elif mutual_check_in:
        intent = "mutual_check_in"
        mode = "converse"
        tone = "warm"

    elif social_check_in:
        intent = "social_check_in"
        mode = "converse"
        tone = "warm"

    elif greeting:
        intent = "greeting"
        mode = "converse"
        tone = "warm"

    elif relationship_question and direct_request:
        intent = "relationship_question"
        mode = "answer"
        tone = "warm"

    elif direct_request:
        intent = "explicit_request"
        mode = "answer"

    elif emotion in {"frustrated", "sad", "worried"} or negative_event:
        intent = "emotional_sharing"
        mode = "support"
        tone = "gentle" if emotion != "frustrated" else "calm"

    elif achievement or emotion == "positive":
        intent = "positive_sharing"
        mode = "acknowledge"
        tone = "warm"

    elif _PERSONAL_UPDATE_PATTERN.search(message):
        intent = "personal_sharing"
        mode = "converse"
        tone = "natural"

    elif emotion == "playful":
        intent = "social_conversation"
        mode = "converse"
        tone = "playful"

    elif uses_recent_context:
        intent = "contextual_follow_up"
        mode = "continue"

    else:
        intent = "general_conversation"
        mode = "converse"

    has_explicit_request = (
        direct_request
        and not greeting
        and not social_check_in
        and not mutual_check_in
        and not (praise_boundary and not direct_request)
    )

    prefer_one_sentence = (
        not has_explicit_request
        and intent in {
            "greeting",
            "social_check_in",
            "mutual_check_in",
            "style_boundary",
            "listening_request",
            "emotional_sharing",
            "positive_sharing",
            "personal_sharing",
            "social_conversation",
            "general_conversation",
            "contextual_follow_up",
        }
    )

    advice_is_unwanted = (
        no_advice
        or explicit_listening
        or praise_boundary
        or (
            intent in {
                "emotional_sharing",
                "positive_sharing",
                "personal_sharing",
                "greeting",
                "social_check_in",
                "mutual_check_in",
                "social_conversation",
                "general_conversation",
            }
            and not has_explicit_request
        )
    )

    avoid_routine_offers = (
        no_routine_help
        or intent in {
            "greeting",
            "social_check_in",
            "mutual_check_in",
            "style_boundary",
            "listening_request",
            "emotional_sharing",
            "positive_sharing",
            "personal_sharing",
            "social_conversation",
            "general_conversation",
        }
    )

    return ConversationDecision(
        intent=intent,
        response_mode=mode,
        emotion=emotion,
        tone=tone,
        prefer_one_sentence=prefer_one_sentence,
        uses_recent_context=uses_recent_context,
        avoid_unsolicited_advice=advice_is_unwanted,
        avoid_routine_help_offers=avoid_routine_offers,
        has_explicit_request=has_explicit_request,
        emotion_confidence=emotion_confidence,
        invites_story_question=(
            intent == "emotional_sharing" and story_opening and negative_event
        ),
    )