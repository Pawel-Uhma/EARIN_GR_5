from test_cases import test_case_1, test_case_2, test_case_3, test_case_4

COLORS = ["Red", "Green", "Blue", "Yellow", "Orange", "Purple", "Pink", "Brown", "Black", "White", "Gray"]


class CSP: 
    def __init__(self, variables, domains, constraints): 
        self.variables = variables 
        self.domains = domains 
        self.constraints = constraints 
        self.solution = None
        self.cur = None

        
    def solve(self): 
        assignment = {} 
        self.solution = self.backtrack(assignment) 
        return self.solution

    def forward_checking(self, var, value, assignment):
        removed = {}
        for neighbor in list(self.constraints[var]):
            if neighbor not in assignment:
                #if the neighbor contains the value, remove it
                if value in self.domains[neighbor]:
                    if neighbor not in removed:
                        removed[neighbor] = []
                    self.domains[neighbor].remove(value)
                    removed[neighbor].append(value)
                    #if the domain was left empty
                    if len(self.domains[neighbor]) == 0:
                        for n in removed:
                            #restore values in domains
                            for v in removed[n]:
                                self.domains[n].append(v)
                        return None
        return removed

    def backtrack(self, assignment):
        #if the assignment is filled, done
        if len(assignment) == len(self.variables):
            return assignment
        var = None
        for v in self.variables:
            if v not in assignment:
                var = v
                break
        for value in list(self.domains[var]): 
            conflict = False
            for neighbor in list(self.constraints[var]):
                #if a neighbor has the same color, try another color
                if neighbor in assignment and assignment[neighbor] == value:
                    conflict = True
                    break
            if conflict:
                continue
            assignment[var] = value
            removed = self.forward_checking(var, value, assignment)
            if removed is not None:
                result = self.backtrack(assignment)
                if result is not None:
                    return result
            # If there is not solution from recursion, delete current assignment
            del assignment[var]
            if removed is not None:
                for neighbor, vals in removed.items():
                    for v in vals:
                        self.domains[neighbor].append(v)
        return None


def solve_case(test_case:dict):
    colors = COLORS.copy()
    # NUMBER OF COLORS TO BE CHECKED IF POSSIBLE
    while True:
        try:
            num_colors = int(input("ENTER NUMBER OF COLORS: "))
            break  
        except ValueError:
            print("Invalid input. Please enter an integer number.")

    if num_colors > len(colors):
        colors = list(range(1,num_colors))

    variables = list(test_case.keys())
    domains = {}
    for var in variables:
        domains[var] = colors[:num_colors]
    constraints = test_case.copy()
    print(f"\nVARIABLES: {variables}\n\nDOMAINS: {domains}\n\nCONSTRAINTS: {constraints}\n")

    csp = CSP(variables, domains, constraints) 
    solution = csp.solve()

    print("Solution: \n", solution)





# solve_case(test_case_1)
# solve_case(test_case_2)
# solve_case(test_case_3)
solve_case(test_case_4)

