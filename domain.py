from enum import Enum, IntEnum
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