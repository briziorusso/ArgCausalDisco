# Mock-Three-Var MUS Encodings

This folder contains two focused encodings to inspect minimal conflicts (MUS) for the mock-three-var scenario.

## Files

- facts_complete_causalaba.lp — Full semantic encoding for the 3-node example; UNSAT with all three facts
- facts_complete_causalaba_adorned.lp — Same encoding with `mus/1` assumptions for MUS over facts; expects one MUS with all three facts

## Quick Checks (aligns with run_tests.sh)

1) Full semantic encoding is UNSAT with all facts

```bash
cd /vol/bitbucket/fr920/ArgCausalDisco-1
clingo encodings/test_lps/mock_three_var_manual/facts_complete_causalaba.lp --quiet=1 --models=0
```

Expected: UNSATISFIABLE.

2) Comment any single fact → SAT

Example (comment last fact `ext_indep(0,1,empty)`):

```bash
cd /vol/bitbucket/fr920/ArgCausalDisco-1
sed 's/^ext_indep(0,1,empty)\./% ext_indep(0,1,empty). (commented)/' \
	encodings/test_lps/mock_three_var_manual/facts_complete_causalaba.lp \
	> /tmp/facts_complete_causalaba_commented.lp
clingo /tmp/facts_complete_causalaba_commented.lp --quiet=1 --models=0
```

Expected: SATISFIABLE (and the same holds if you comment any of the three facts; models change depending on which fact is removed).

3) Semantic MUS over `mus/1` assumptions (all three facts in the core)

```bash
cd /vol/bitbucket/fr920/ArgCausalDisco-1/encodings/test_lps/mock_three_var_manual
clingo facts_complete_causalaba_adorned.lp --output=smodels | \
	/vol/bitbucket/fr920/wasp/build/release/wasp --mus=mus -n 0
```

Expected: exactly one MUS, containing all three facts (mus(1), mus(2), mus(3)).

## Which file to use

- facts_complete_causalaba.lp: baseline semantic encoding to check SAT/UNSAT of the full program.
- facts_complete_causalaba_adorned.lp: same encoding with assumptions for MUS over facts (mus/1).

## Notes

- Avoid using o() predicates as these seem to conflict with other parts of the program
- Avoid using #show statements as this impede printing of the predicate in a MUS
