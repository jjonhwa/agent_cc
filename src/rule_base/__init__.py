from .anger import rule_anger
from .economy import rule_economy
from .family import rule_family
from .impulse import rule_impulse
from .isolation import rule_isolation
from .risk import rule_risk
from .social import rule_social

RULE_REGISTRY: dict = {
    "Family": rule_family,
    "Anger": rule_anger,
    "Economy": rule_economy,
    "Impulse": rule_impulse,
    "Social": rule_social,
    "Isolation": rule_isolation,
    # Risk returns int (0–3) instead of bool
    "Risk": rule_risk,
}
