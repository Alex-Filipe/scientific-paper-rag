from paper_rag.numbers import answer_numbers, claimed_readings


def test_answer_numbers_skip_citations_and_references() -> None:
    assert answer_numbers("O erro é 4,49% [1], como em [10, 41]; 4,49 de novo.") == ["4,49"]


def test_claimed_readings_leave_out_numbers_from_the_question() -> None:
    claimed = claimed_readings("O ResNet-152 chega a 4,49% [2] com 50k imagens.", "ResNet-152?")

    assert claimed == {4.49, 50.0, 50_000.0}
