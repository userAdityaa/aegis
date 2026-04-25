from .account_takeover import AccountTakeoverAttack
from .base import AttackInjector, BaseAttack
from .cicd_poisoning import CICDPoisoningAttack
from .dead_drop_hijack import DeadDropHijackAttack
from .dependency_confusion import DependencyConfusionAttack
from .long_con import LongConAttack
from .metadata_injection import MetadataInjectionAttack
from .star_jacking import StarJackingAttack
from .typosquatting import TyposquattingAttack

ALL_ATTACKS = (
	TyposquattingAttack,
	LongConAttack,
	AccountTakeoverAttack,
	CICDPoisoningAttack,
	DependencyConfusionAttack,
	MetadataInjectionAttack,
	StarJackingAttack,
	DeadDropHijackAttack,
)

__all__ = [
	"ALL_ATTACKS",
	"AccountTakeoverAttack",
	"AttackInjector",
	"BaseAttack",
	"CICDPoisoningAttack",
	"DeadDropHijackAttack",
	"DependencyConfusionAttack",
	"LongConAttack",
	"MetadataInjectionAttack",
	"StarJackingAttack",
	"TyposquattingAttack",
]