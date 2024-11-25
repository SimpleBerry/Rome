def create_mctsr():
    return MCTSr()

def create_tot():
    return TOT()


NAME2SOLVER = {
    "mctsr": create_mctsr(),
    "tot": create_tot(),
}
