from karpo_decision.train import order as od


def choice(criteria, instructions="Which one fits?"):
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def test_what_never_moves():
    cases = [
        choice({"A": "cats", "B": "dogs", "C": "all of the above"}),
        choice({"A": "cats", "B": "dogs", "C": "both A and B"}),
        choice({"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}),
        choice({"A": "1 star", "B": "2 stars", "C": "3 stars"}),
        choice({"A": "low", "B": "medium", "C": "high"}),
        choice({"A": "cats", "B": "dogs"}, "Pick the first option unless the text says otherwise."),
        choice({"A": "only one"}),
        {"type": "score", "instructions": "Rate it.", "criteria": ["bad", "ok", "good"]},
        {"type": "noul", "instructions": "Is it true?"},
    ]
    for q in cases:
        assert not od.movable(q), q


def test_only_capital_letters_standing_alone_point():
    for text in ("Buy tickets for a showing", "Ford Focus Wagon", "has part(s)", "Only a few seats remain",
                 "core values",
                 "A is a precondition for B", "Solve for x (x) first", "the former employee", "B12 and folate"):
        assert not od.POINTING.search(text), text
    for text in ("All of the above", "none of these", "Both A and C", "A, B and D", "(B)", "B only", "Only C",
                 "option 2", "the former"):
        assert od.POINTING.search(text), text
    assert od.movable(choice({"A": "Check whether tickets are available for a showing",
                              "B": "Buy tickets for a showing", "C": "Reply to the user"}))


def test_lower_case_letters_point_where_nothing_else_can_be_meant():
    for text in ("both a and b", "Both (b) and (c)", "either a or c", "neither a nor b", "a & b", "a, b, c",
                 "(a) and (b)",
                 "b,c,d."):
        assert od.POINTING.search(text), text
    for text in ("Fe(s) + CuSO4(aq) -> Cu(s)", "gram (g)", "a violation of FRE 404(b)", "vitamin a and c",
                 "holding that subparagraphs a and b are mutually exclusive",
                 "Assign variables a, b, and c to the sides"):
        assert not od.POINTING.search(text), text


def test_a_stem_that_writes_its_options_out_never_moves():
    q = choice({"A": "Neon gas", "B": "Oxygen gas", "C": "Chlorine gas"},
               "Which gas is inert?\n\nA: Neon gas\nB: Oxygen gas")
    assert not od.movable(q)
    assert od.movable(choice({"A": "Neon gas", "B": "Oxygen gas"}, "Which of neon gas and oxygen gas is inert?"))


def test_by_place_brackets_are_option_letters():
    assert od.BY_PLACE.search("Pick (B) unless the text says otherwise.")
    assert not od.BY_PLACE.search("Which product forms, (R)-1-phenylethanol or its enantiomer?")


def test_computed_numbers_move():
    assert od.movable(choice({"A": "$345", "B": "$12", "C": "$3,400"}))
    assert od.movable(choice({"A": "23", "B": "7", "C": "101"}))
