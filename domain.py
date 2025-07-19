from enum import Enum, IntEnum
from typing import Tuple, Final

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
    @classmethod
    def from_demographic(cls, demographic: str) -> 'AgeRating':
        """새로운 데이터 형식의 demographics에서 연령 등급 추출"""
        demographic = demographic.lower()
        
        if demographic == 'shounen':
            return cls.YOUTH    # 12세 이상
        elif demographic == 'shoujo':
            return cls.YOUTH    # 12세 이상
        elif demographic == 'seinen':
            return cls.ADULT    # 18세 이상
        elif demographic == 'josei':
            return cls.ADULT    # 18세 이상
        else:
            return cls.ALL      # 기본값: 전체연령

# Demographic 열거형 
class Demographic(Enum):
    SHOUNEN = "Shounen"      # 소년 대상
    SHOUJO = "Shoujo"        # 소녀 대상
    SEINEN = "Seinen"        # 성인 남성 대상
    JOSEI = "Josei"          # 성인 여성 대상
    KIDS = "Kids"            # 어린이 대상s
    
    @classmethod
    def from_age_and_gender(cls, age_group: str, gender: str) -> 'Demographic':
        """연령대와 성별을 기반으로 적절한 demographic 매핑"""
        
        # 어린이 대상 (12세 미만)
        if age_group == "12~15":
            return cls.KIDS
        
        # 청소년 대상 (15-17세)
        elif age_group == "15~18":
            if gender == "male":
                return cls.SHOUNEN  # 소년 대상
            elif gender == "female":
                return cls.SHOUJO   # 소녀 대상
            else:
                return cls.SHOUNEN  # 기본값
        
        # 성인 대상 (18세 이상)
        else:
            if gender == "male":
                return cls.SEINEN   # 성인 남성 대상
            elif gender == "female":
                return cls.JOSEI    # 성인 여성 대상
            else:
                return cls.SEINEN   # 기본값