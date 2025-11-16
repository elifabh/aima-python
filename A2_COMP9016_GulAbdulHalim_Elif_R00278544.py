import sys
sys.path.append('./aima-python')

from logic import FolKB, expr, fol_fc_ask, fol_bc_ask,Expr
from probability import BayesNet, enumeration_ask

import numpy as np
import pandas as pd
from learning import NaiveBayesLearner, DataSet
from collections import Counter


# --------------------------------------------------------------------
# ----------------------1.1	LOGICAL REASONING-------------------------
# --------------------------------------------------------------------

def q1_logical_reasoning_mtu():
    """"
    # ---------------------1.1.1 DEFINE THE KNOWLEDGE BASE-------------
    1. Predicates (FOL domain)

        Student(s)      : s is a student
        Lecturer(l)     : l is a lecturer
        Module(m)       : m is a module
        Takes(s, m)     : student s is currently taking module m
        Passed(s, m)    : student s has already passed module m
        Teaches(l, m)   : lecturer l teaches module m
        Prereq(p, m)    : module p is a (direct) prerequisite for m
        Eligible(s, m)  : s has satisfied all prerequisites for m
        CanEnroll(s, m) : s is eligible and has not already passed m
        TaughtBy(s, l)  : s is taught by lecturer l
        Classmate(s, t) : students s and t share a module and s != t
        NotSame(s, t)   : s and t are different individuals  
        NotPassed(s, m) : s has not passed module m 
        IndirectPrereq(p, m): p is an (in)direct prerequisite of m

    2.  Logical Rules in FOL

        1) Teaching implies "taught by" when a student is enrolled:
        ∀s,l,m  (Takes(s,m) ∧ Teaches(l,m)) → TaughtBy(s,l)

        2)Two students are classmates if they take the same module and are not the same person:
        ∀s,t,m: (Takes(s,m) ∧ Takes(t,m) ∧ NotSame(s,t)) → Classmate(s,t)

        3)A student is eligible for m if they have passed all prerequisites:
         ∀s,p,m: (Prereq(p,m) ∧ Passed(s,p)) → Eligible(s,m)

         4)A student can enrol in a module if they are eligible and have not already passed it:
         ∀s,m (Eligible(s,m) ∧ ¬Passed(s,m)) → CanEnroll(s,m)

        5) Prerequisites are transitive:
        ∀p,m,n (Prereq(p,m) ∧ Prereq(m,n)) → Prereq(p,n)

    # ------------------------- 1.1.2 PROVIDE THE FACTS -----------------

    1.  Adding facts
    
        Modules:   COMP9016, COMP9062, COMP9061, COMP9058
        Students:  Alice, Bob, Eve
        Lecturers: DrDan, DrSophie, DrLisa

        Prerequisites:
            Prereq(COMP9016, COMP9062)
            Prereq(COMP9062, COMP9061)

        Teaching:
            Teaches(DrDan, COMP9016)
            Teaches(DrSophie, COMP9062)
            Teaches(DrLisa, COMP9061)

        Student Record:
            Passed(Alice, COMP9016), Passed(Alice, COMP9058)
            Passed(Bob, COMP9016),   Passed(Bob, COMP9062)

        Current Enrolments:
            Takes(Alice, COMP9062)
            Takes(Bob, COMP9061)
            Takes(Eve, COMP9016)

    # ------------------------- 1.1.3 INFERENCE AND ANALYSIS -----------------
    1.  Forward Chaining (FOL-FC)

            fol_fc_ask(kb, expr("TaughtBy(s,l)"))
            fol_fc_ask(kb, expr("IndirectPrereq(p,m)"))
    
    2.  Backward Chaining (FOL-BC)

            Eligible(Alice,COMP9061)
            Eligible(Bob, COMP9061)
            TaughtBy(Eve,l)
            Classmate(Alice, Eve)  (before and after adding Takes(Alice, COMP9016))

        """


# creating the knowledge base 
    kb = FolKB()

    # ---------------------1.1.1 DEFINE THE KNOWLEDGE BASE-------------
    
    # RULES
    # 1) TaughtBy: Teaching implies "taught by" when student is enrolled
    kb.tell(expr("(Takes(s,m) & Teaches(l,m)) ==> TaughtBy(s,l)"))
    
    # 2) Classmate: Two students are classmates if they take the same module
    kb.tell(expr("(Student(s) & Student(t) & Takes(s,m) & Takes(t,m) & NotSame(s,t)) ==> Classmate(s,t)"))
    
    # 3) Prereq -> IndirectPrereq (direct prerequisites are also indirect)
    kb.tell(expr("Prereq(p,m) ==> IndirectPrereq(p,m)"))
    
    # 4) Transitivity: Prerequisites are transitive
    kb.tell(expr("(Prereq(p,m) & IndirectPrereq(m,n)) ==> IndirectPrereq(p,n)"))
    
    # 5) Eligibility: Student is eligible if passed all prerequisites
    kb.tell(expr("Passed(s, COMP9016) ==> Eligible(s, COMP9062)"))
    kb.tell(expr("(Passed(s, COMP9016) & Passed(s, COMP9062)) ==> Eligible(s, COMP9061)"))
    
    # 6) CanEnroll: Student can enroll if eligible and not already passed
    kb.tell(expr("(Eligible(s,m) & NotPassed(s,m)) ==> CanEnroll(s,m)"))
    

    # ---------------------1.1.2 PROVIDE THE FACTS---------------------

    # Students
    for student in ["Alice", "Bob", "Eve"]:
        kb.tell(expr(f"Student({student})"))
    
    # Lecturers
    for lecturer in ["DrDan", "DrSophie", "DrLisa"]:
        kb.tell(expr(f"Lecturer({lecturer})"))
    
    # Modules
    for module in ["COMP9016", "COMP9062", "COMP9061", "COMP9058"]:
        kb.tell(expr(f"Module({module})"))
    
    # Prerequisites
    kb.tell(expr("Prereq(COMP9016, COMP9062)"))
    kb.tell(expr("Prereq(COMP9062, COMP9061)"))
    
    # Teaching
    kb.tell(expr("Teaches(DrDan, COMP9016)"))
    kb.tell(expr("Teaches(DrSophie, COMP9062)"))
    kb.tell(expr("Teaches(DrLisa, COMP9061)"))
    
    # Student Record: Passed
    kb.tell(expr("Passed(Alice, COMP9016)"))
    kb.tell(expr("Passed(Alice, COMP9058)"))
    kb.tell(expr("Passed(Bob, COMP9016)"))
    kb.tell(expr("Passed(Bob, COMP9062)"))
    
    # Current Enrolments: Takes
    kb.tell(expr("Takes(Alice, COMP9062)"))
    kb.tell(expr("Takes(Bob, COMP9061)"))
    kb.tell(expr("Takes(Eve, COMP9016)"))
    
    # NotSame facts (for Classmate rule)
    students = ["Alice", "Bob", "Eve"]
    for s in students:
        for t in students:
            if s != t:
                kb.tell(expr(f"NotSame({s}, {t})"))
    
    # NotPassed facts (for CanEnroll rule )
    kb.tell(expr("NotPassed(Alice, COMP9062)"))
    kb.tell(expr("NotPassed(Alice, COMP9061)"))
    kb.tell(expr("NotPassed(Bob, COMP9061)"))
    kb.tell(expr("NotPassed(Eve, COMP9016)"))
    
    print("✓ Knowledge Base initialized\n")
    

    # ---------------------1.1.3 INFERENCE AND ANALYSIS----------------
    

    # FORWARD CHAINING
    print("="*60)
    print("FORWARD CHAINING (FC) ")
    print("="*60)
    
    # Query 1: TaughtBy(s,l)
    print("\n[FC] Query: TaughtBy(s,l)")
    fc_taughtby = list(fol_fc_ask(kb, expr("TaughtBy(s,l)")))
    print(f"Results: {fc_taughtby}")
    
    # Query 2: IndirectPrereq(p,m)
    print("\n[FC] Query: IndirectPrereq(p,m)")
    fc_indirect = list(fol_fc_ask(kb, expr("IndirectPrereq(p,m)")))
    print(f"Results: {fc_indirect}")
    
    # BACKWARD CHAINING
    print("\n" + "="*60)
    print("BACKWARD CHAINING (BC)")
    print("="*60)
    
    # Query 1: Eligible(Alice, COMP9061)
    print("\n[BC] Query: Eligible(Alice, COMP9061)")
    bc_alice = list(fol_bc_ask(kb, expr("Eligible(Alice, COMP9061)")))
    print(f"Result: {bc_alice}")
    
    # Query 2: Eligible(Bob, COMP9061)
    print("\n[BC] Query: Eligible(Bob, COMP9061)")
    bc_bob = list(fol_bc_ask(kb, expr("Eligible(Bob, COMP9061)")))
    print(f"Result: {bc_bob}")
    
    # Query 3: TaughtBy(Eve, l)
    print("\n[BC] Query: TaughtBy(Eve, l)")
    bc_eve = list(fol_bc_ask(kb, expr("TaughtBy(Eve, l)")))
    print(f"Result: {bc_eve}")
    
    # Query 4: Classmate(Alice, Eve) - BEFORE
    print("\n[BC] Query: Classmate(Alice, Eve) BEFORE Takes(Alice, COMP9016)")
    bc_classmate_before = list(fol_bc_ask(kb, expr("Classmate(Alice, Eve)")))
    print(f"Result: {bc_classmate_before}")
    
    # Add new fact
    kb.tell(expr("Takes(Alice, COMP9016)"))
    print("\n>>> Added fact: Takes(Alice, COMP9016)")
    
    # Query 4: Classmate(Alice, Eve) - AFTER
    print("\n[BC] Query: Classmate(Alice, Eve) AFTER Takes(Alice, COMP9016)")
    bc_classmate_after = list(fol_bc_ask(kb, expr("Classmate(Alice, Eve)")))
    print(f"Result: {bc_classmate_after}")
    

# --------------------------------------------------------------------
# -------------1.2 BAYESIAN NETWORKS: AI MARKET ANALYSIs---------------
# --------------------------------------------------------------------

def q2_bayesian_network_ai_market():
    """
    Question 1.2: Bayesian Networks - AI Market Analysis

    Boolean variables (semantics):
        HypeLevel          : True = high hype
        EnterpriseAdoption : True = high enterprise AI adoption
        VCInvestment       : True = high VC funding
        ComputeCosts       : True = low compute costs
        LabourMarketImpact : True = high labour-market impact
        RegulatoryPressure : True = high regulatory pressure
    """

    from probability import BayesNet, enumeration_ask

    # --------------------- 1.2.1 DEFINE THE BAYES NET ---------------------

    ai_market = BayesNet([
        # HypeLevel (no parents) – prior
        ('HypeLevel', '',
         0.5),  # P(HypeLevel=True)

        # EnterpriseAdoption | HypeLevel
        ('EnterpriseAdoption', 'HypeLevel',
         {True: 0.8,   # P(E=True | H=True)
          False: 0.3}  # P(E=True | H=False)
         ),

        # VCInvestment | EnterpriseAdoption
        ('VCInvestment', 'EnterpriseAdoption',
         {True: 0.8,   # P(V=True | E=True)
          False: 0.2}  # P(V=True | E=False)
         ),

        # ComputeCosts | VCInvestment
        # True = low cost
        ('ComputeCosts', 'VCInvestment',
         {True: 0.7,   # P(C=True (low) | V=True)
          False: 0.3}  # P(C=True (low) | V=False)
         ),

        # LabourMarketImpact | EnterpriseAdoption
        ('LabourMarketImpact', 'EnterpriseAdoption',
         {True: 0.8,   # P(L=True | E=True)
          False: 0.2}  # P(L=True | E=False)
         ),

        # RegulatoryPressure | HypeLevel, EnterpriseAdoption, VCInvestment
        # Parents order: 'HypeLevel EnterpriseAdoption VCInvestment'
        ('RegulatoryPressure', 'HypeLevel EnterpriseAdoption VCInvestment', {
            (True,  True,  True):  0.90,
            (True,  True,  False): 0.75,
            (True,  False, True):  0.70,
            (True,  False, False): 0.50,
            (False, True,  True):  0.70,
            (False, True,  False): 0.50,
            (False, False, True):  0.40,
            (False, False, False): 0.20
        }),
    ])

    print("✓ Bayesian Network initialised\n")

    # --------------------- 1.2.2 INFERENCE QUERIES ------------------------

    print("=" * 60)
    print("INFERENCE QUERIES (ENUMERATION)")
    print("=" * 60)

    # Q1: P(EnterpriseAdoption | HypeLevel=True)
    print("\n[Q1] P(EnterpriseAdoption | HypeLevel=True):")
    q1 = enumeration_ask('EnterpriseAdoption',
                         {'HypeLevel': True},
                         ai_market)
    print("  ", q1.show_approx())

    # Q2: P(RegulatoryPressure | H=True, E=True, V=True)
    print("\n[Q2] P(RegulatoryPressure | HypeLevel=True, EnterpriseAdoption=True, VCInvestment=True):")
    q2 = enumeration_ask('RegulatoryPressure',
                         {'HypeLevel': True,
                          'EnterpriseAdoption': True,
                          'VCInvestment': True},
                         ai_market)
    print("  ", q2.show_approx())

    # Q3: P(ComputeCosts | HypeLevel=True)  (bu, E,V,L,R üzerinden toplama yapar)
    print("\n[Q3] P(ComputeCosts | HypeLevel=True):")
    q3 = enumeration_ask('ComputeCosts',
                         {'HypeLevel': True},
                         ai_market)
    print("  ", q3.show_approx())

    # --------------------- 1.2.3 CONDITIONAL INDEPENDENCE DEMOS ----------

    print("\n" + "=" * 60)
    print("CONDITIONAL INDEPENDENCE (d-separation demos)")
    print("=" * 60)

    # Chain: HypeLevel → EnterpriseAdoption → VCInvestment
    print("\n[Chain] HypeLevel → EnterpriseAdoption → VCInvestment")
    print("Compare P(VCInvestment | H=True) vs P(VCInvestment | H=True, E=True)")

    chain1 = enumeration_ask('VCInvestment',
                             {'HypeLevel': True},
                             ai_market)
    print("  P(VCInvestment | H=True):")
    print("  ", chain1.show_approx())

    chain2 = enumeration_ask('VCInvestment',
                             {'HypeLevel': True,
                              'EnterpriseAdoption': True},
                             ai_market)
    print("  P(VCInvestment | H=True, E=True):")
    print("  ", chain2.show_approx())

    # Fork: EnterpriseAdoption → {VCInvestment, LabourMarketImpact}
    print("\n[Fork] EnterpriseAdoption → {VCInvestment, LabourMarketImpact}")
    print("Compare P(VCInvestment | E=True) vs P(VCInvestment | E=True, L=True)")

    fork1 = enumeration_ask('VCInvestment',
                            {'EnterpriseAdoption': True},
                            ai_market)
    print("  P(VCInvestment | E=True):")
    print("  ", fork1.show_approx())

    fork2 = enumeration_ask('VCInvestment',
                            {'EnterpriseAdoption': True,
                             'LabourMarketImpact': True},
                            ai_market)
    print("  P(VCInvestment | E=True, L=True):")
    print("  ", fork2.show_approx())

    # Collider: {HypeLevel, EnterpriseAdoption, VCInvestment} → RegulatoryPressure
    print("\n[Collider] {HypeLevel, EnterpriseAdoption, VCInvestment} → RegulatoryPressure")
    print("Compare P(HypeLevel | E=True) vs P(HypeLevel | E=True, R=True)")

    col1 = enumeration_ask('HypeLevel',
                           {'EnterpriseAdoption': True},
                           ai_market)
    print("  P(HypeLevel | E=True):")
    print("  ", col1.show_approx())

    col2 = enumeration_ask('HypeLevel',
                           {'EnterpriseAdoption': True,
                            'RegulatoryPressure': True},
                           ai_market)
    print("  P(HypeLevel | E=True, R=True):")
    print("  ", col2.show_approx())

    print("\n✓ Bayesian Network queries completed")


# --------------------------------------------------------------------
# -----1.3 IMPLEMENTATION AND ANALYSIS OF NAIVE BAYES CLASSIFIERS-----
# --------------------------------------------------------------------


if __name__ == "__main__":
    #q1_logical_reasoning_mtu()
    q2_bayesian_network_ai_market()

