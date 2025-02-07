from src.data_model.identifiers.identifier import Identifier
from src.data_model.identifiers.user.user_identifier import UserIdentifier
from src.data_model.identifiers.instrument.instrument_identifier import (
    InstrumentIdentifier,
)


def main():
    user_01_id = UserIdentifier("user_01")
    inst_ID_001 = InstrumentIdentifier("actigraph", "0001")
    print(user_01_id)
    print(inst_ID_001)


if __name__ == "__main__":
    main()
