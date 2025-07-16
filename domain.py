from enum import Enum, IntEnum
from typing import Tuple, Final

# 성별 열거형
class Gender(Enum):
    MALE = "male"
    FEMALE = "female"
    SKIP = "skip"
    
    @classmethod
    def from_option(cls, option: str) -> 'Gender':
        mapping = {
            "남": cls.MALE,
            "여": cls.FEMALE,
            "넘어가기": cls.SKIP
        }
        return mapping.get(option, cls.SKIP)

# 연령대 열거형  
class AgeGroup(Enum):
    TEEN_12_15 = (12, 15)
    TEEN_15_18 = (15, 18)
    ADULT_18_30 = (18, 30)
    ADULT_30_40 = (30, 40)
    ADULT_40_50 = (40, 50)
    ADULT_50_PLUS = (50, 100)
    
    @classmethod
    def from_option(cls, option: str) -> 'AgeGroup':
        mapping = {
            "12~15": cls.TEEN_12_15,
            "15~18": cls.TEEN_15_18,
            "18~30": cls.ADULT_18_30,
            "30~40": cls.ADULT_30_40,
            "40~50": cls.ADULT_40_50,
            "50~": cls.ADULT_50_PLUS
        }
        return mapping.get(option, cls.ADULT_18_30)
    
    @property
    def min_age(self) -> int:
        return self.value[0]
    
    @property
    def max_age(self) -> int:
        return self.value[1]

# 연령 등급 열거형
class AgeRating(IntEnum):
    ALL_AGES = 0
    OVER_12 = 12
    OVER_15 = 15
    OVER_18 = 18
    UNKNOWN = 0
    
    @classmethod
    def from_option(cls, option: str) -> 'AgeRating':
        mapping = {
            "전체연령": cls.ALL_AGES,
            "12세 이상": cls.OVER_12,
            "15세 이상": cls.OVER_15,
            "18세 이상": cls.OVER_18,
            "확인필요": cls.UNKNOWN
        }
        return mapping.get(option, cls.ALL_AGES)
