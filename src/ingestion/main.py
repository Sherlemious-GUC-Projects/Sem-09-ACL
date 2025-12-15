### ~~~ GLOBAL IMPORTS ~~~ ###
# None

### ~~~ LOCAL IMPORTS ~~~ ###
from utils.types import IntentType, Entity, ProcessedQuery
from ingestion import extract_ner as ner, analyise_sentiment as sentiment
from utils.constant import CSV_PATH


def main(raw_query: str) -> ProcessedQuery:
    """"""
    #################################
    ### ~~~ NER INGESTION RUN ~~~ ###
    #################################

    ### 0. ensure nltk resources ###
    ner._ensure_resources()

    ### 1. load refrence data ###
    reference_data: ner.ReferenceData = ner.load_reference_data(
        CSV_PATH,
    )

    ### 2. run NER extraction ###
    entities: list[Entity] = ner.extract_ner(
        raw_query,
        reference_data,
    )

    ###################################
    ### ~~~ INTENT ANALYSIS RUN ~~~ ###
    ###################################

    ### 1. run intent analysis ###
    intent: IntentType = sentiment.analyze_intent(
        raw_query,
    )

    #####################################
    ### ~~~ PACKAGE INGESTION RUN ~~~ ###
    #####################################

    processed_query = ProcessedQuery(
        original_text=raw_query,
        intent=intent,
        entities=entities,
    )

    return processed_query


if __name__ == "__main__":
    test = (
        ProcessedQuery(
            original_text=(
                "I need options for getting to London next Tuesday "
                "that land before noon at LHR."
            ),
            intent=IntentType.FLIGHT_SEARCH,
            entities=[
                Entity("DATE", "next Tuesday"),
                Entity("AIRPORT", "LHR"),
            ],
        ),
    )
    outcome = main(test[0].original_text)
    valid: bool = False
    if outcome.intent == test[0].intent:
        if len(outcome.entities) == len(test[0].entities):
            match_count = 0
            for ent in outcome.entities:
                for test_ent in test[0].entities:
                    if (
                        ent.entity_type == test_ent.entity_type
                        and ent.value == test_ent.value
                    ):
                        match_count += 1
                        break
            if match_count == len(test[0].entities):
                valid = True
    print("Test passed:", valid)
    from src.utils.types import pretty_print

    pretty_print(outcome)
    pretty_print(test[0])
#    __   _,_ /_ __,
#  _(_/__(_/_/_)(_/(_
#   _/_
#  (/
