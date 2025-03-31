#TEST CASE 1
test_case_1 = {
    "ab": ["bc", "nt", "sk"],
    "bc": ["yt", "nt", "ab"],
    "mb": ["sk", "nu", "on"],
    "nb": ["qc", "ns", "pe"],
    "ns": ["nb", "pe"],
    "nl": ["qc"],
    "nt": ["bc", "yt", "ab", "sk", "nu"],
    "nu": ["nt", "mb"],
    "on": ["mb", "qc"],
    "pe": ["nb", "ns"],
    "qc": ["on", "nb", "nl"],
    "sk": ["ab", "mb", "nt"],
    "yt": ["bc", "nt"]
}



#TEST CASE 2: UNITED STATES
test_case_2 = {
    "AL": ["FL", "GA", "TN", "MS"],
    "AK": [],  
    "AZ": ["CA", "NV", "UT", "CO", "NM"],
    "AR": ["MO", "TN", "MS", "LA", "TX", "OK"],
    "CA": ["OR", "NV", "AZ"],
    "CO": ["WY", "NE", "KS", "OK", "NM", "AZ", "UT"],
    "CT": ["NY", "MA", "RI"],
    "DE": ["MD", "PA"], 
    "FL": ["AL", "GA"],
    "GA": ["FL", "AL", "TN", "NC", "SC"],
    "HI": [],  
    "ID": ["WA", "OR", "NV", "UT", "WY", "MT"],
    "IL": ["WI", "IA", "MO", "KY", "IN"],
    "IN": ["MI", "OH", "KY", "IL"],
    "IA": ["MN", "WI", "IL", "MO", "NE", "SD"],
    "KS": ["NE", "MO", "OK", "CO"],
    "KY": ["IL", "IN", "OH", "WV", "VA", "TN", "MO"],
    "LA": ["TX", "AR", "MS"],
    "ME": ["NH"],
    "MD": ["PA", "DE", "WV", "VA"],
    "MA": ["NY", "VT", "NH", "CT", "RI"],
    "MI": ["OH", "IN", "WI"],
    "MN": ["ND", "SD", "IA", "WI"],
    "MS": ["LA", "AR", "TN", "AL"],
    "MO": ["IA", "IL", "KY", "TN", "AR", "OK", "KS", "NE"],
    "MT": ["ND", "SD", "WY", "ID"],
    "NE": ["SD", "IA", "MO", "KS", "CO", "WY"],
    "NV": ["OR", "ID", "UT", "AZ", "CA"],
    "NH": ["ME", "MA", "VT"],
    "NJ": ["NY", "PA", "DE"],
    "NM": ["AZ", "CO", "OK", "TX", "UT"],
    "NY": ["PA", "NJ", "CT", "MA", "VT"],
    "NC": ["VA", "TN", "GA", "SC"],
    "ND": ["MN", "SD", "MT"],
    "OH": ["PA", "WV", "KY", "IN", "MI"],
    "OK": ["KS", "MO", "AR", "TX", "NM", "CO"],
    "OR": ["WA", "ID", "NV", "CA"],
    "PA": ["NY", "NJ", "DE", "MD", "WV", "OH"],
    "RI": ["MA", "CT"],
    "SC": ["NC", "GA"],
    "SD": ["ND", "MN", "IA", "NE", "WY", "MT"],
    "TN": ["KY", "VA", "NC", "GA", "AL", "MS", "AR", "MO"],
    "TX": ["NM", "OK", "AR", "LA"],
    "UT": ["ID", "WY", "CO", "NM", "AZ", "NV"],
    "VT": ["NY", "MA", "NH"],
    "VA": ["MD", "NC", "TN", "KY", "WV"],
    "WA": ["OR", "ID"],
    "WV": ["OH", "PA", "MD", "VA", "KY"],
    "WI": ["MN", "IA", "IL", "MI"],
    "WY": ["MT", "SD", "NE", "CO", "UT", "ID"]
}

#TEST CASE 3: TRIANGLE
test_case_3 = {
    "A": ["B", "C"],
    "B": ["A", "C"],
    "C": ["A", "B"]
}

#TEST CASE 4: LINEAR CHAIN
test_case_4 = {
    "A": ["B"],
    "B": ["A", "C"],
    "C": ["B", "D"],
    "D": ["C"]
}
