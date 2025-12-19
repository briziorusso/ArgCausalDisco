#!/bin/bash
# Manual test script for mock_three_var MUS analysis

echo "============================================"
echo "Manual MUS Testing for mock_three_var"
echo "============================================"
echo ""

cd /vol/bitbucket/fr920/ArgCausalDisco-1

echo "TEST 1: Full CausalABA (all facts)"
echo "-------------------------------------"
echo "Running: clingo facts_complete_causalaba.lp --models=0"
echo ""
clingo encodings/test_lps/mock_three_var_manual/facts_complete_causalaba.lp \
  --quiet=1 --models=0 2>&1 | grep -E "SATISFIABLE|UNSATISFIABLE|Models"
echo ""
echo "Expected: UNSATISFIABLE with all three facts present"
echo ""

echo "TEST 2: Full CausalABA with last fact commented"
echo "-------------------------------------"
echo "Commenting out the last fact (ext_indep(0,1,empty)) and rerunning"
echo ""
sed 's/^ext_indep(0,1,empty)\./% ext_indep(0,1,empty). (commented)/' \
  encodings/test_lps/mock_three_var_manual/facts_complete_causalaba.lp \
  > /tmp/facts_complete_causalaba_commented.lp

clingo /tmp/facts_complete_causalaba_commented.lp --quiet=1 --models=0 2>&1 | \
  grep -E "SATISFIABLE|UNSATISFIABLE|Models"
echo ""
echo "Expected: SATISFIABLE (removing/commenting any one fact makes it SAT; models differ per removal)"
echo ""

echo "TEST 3: Semantic MUS with adorned assumptions"
echo "-------------------------------------"
echo "Running: clingo facts_complete_causalaba_adorned.lp --output=smodels | wasp --mus=mus -n 0"
echo ""
clingo encodings/test_lps/mock_three_var_manual/facts_complete_causalaba_adorned.lp --output=smodels 2>/dev/null | \
  /vol/bitbucket/fr920/wasp/build/release/wasp --mus=mus -n 0 2>&1 | grep -E "\[MUS|Total|Optimum"
echo ""
echo "Expected: a single MUS containing all three mus(i) facts (all three input facts)"
echo ""

echo "============================================"
echo "Summary"
echo "============================================"
echo "- facts_complete_causalaba.lp is UNSAT with all three facts."
echo "- Commenting/removing any single fact makes it SAT (different models per removal)."
echo "- facts_complete_causalaba_adorned.lp yields exactly one MUS over mus/1 containing all three facts."
echo ""
