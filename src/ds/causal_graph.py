import itertools
import re
import random
from copy import deepcopy
from collections import deque


class CausalGraph:
    def __init__(self, V, directed_edges=[], bidirected_edges=[]):
        self.de = directed_edges
        self.be = bidirected_edges

        self.v = list(V)
        self._set_v = set(V)
        self.pa = {v: set() for v in V} # parents (directed edges)
        self.ch = {v: set() for v in V} # children (directed edges)
        self.ne = {v: set() for v in V} # neighbors (bidirected edges)
        self.bi = set(map(tuple, map(sorted, bidirected_edges))) # bidirected edges

        for v1, v2 in directed_edges:
            self.pa[v2].add(v1)
            self.ch[v1].add(v2)

        for v1, v2 in bidirected_edges:
            self.ne[v1].add(v2)
            self.ne[v2].add(v1)
            self.bi.add(tuple(sorted((v1, v2))))

        self.pa = {v: sorted(self.pa[v]) for v in self.v}
        self.ch = {v: sorted(self.ch[v]) for v in self.v}
        self.ne = {v: sorted(self.ne[v]) for v in self.v}

        self._sort()
        self.v2i = {v: i for i, v in enumerate(self.v)}

        self.cc = self._c_components()
        self.v2cc = {v: next(c for c in self.cc if v in c) for v in self.v}
        self.pap = {
            v: sorted(set(itertools.chain.from_iterable(
                self.pa[v2] + [v2]
                for v2 in self.v2cc[v]
                if self.v2i[v2] <= self.v2i[v])) - {v},
                key=self.v2i.get)
            for v in self.v}
        self.c2 = self._maximal_cliques()
        self.v2c2 = {v: [c for c in self.c2 if v in c] for v in self.v}

    def __iter__(self):
        return iter(self.v)

    def subgraph(self, V_sub):
        assert V_sub.issubset(self._set_v)

        new_de = [(V1, V2) for V1, V2 in self.de if V1 in V_sub and V2 in V_sub]
        new_be = [(V1, V2) for V1, V2 in self.be if V1 in V_sub and V2 in V_sub]

        return CausalGraph(V_sub, new_de, new_be)
    
    def _sort(self): # sort V topologically
        L = []
        marks = {v: 0 for v in self.v}

        def visit(v):
            if marks[v] == 2:
                return
            if marks[v] == 1:
                raise ValueError('Not a DAG.')

            marks[v] = 1
            for c in self.ch[v]:
                visit(c)
            marks[v] = 2
            L.append(v)

        for v in marks:
            if marks[v] == 0:
                visit(v)
        self.v = L[::-1]

    def _c_components(self):
        pool = set(self.v)
        cc = []
        while pool:
            cc.append({pool.pop()})
            while True:
                added = {k2 for k in cc[-1] for k2 in self.ne[k]}
                delta = added - cc[-1]
                cc[-1].update(delta)
                pool.difference_update(delta)
                if not delta:
                    break
        return [tuple(sorted(c, key=self.v2i.get)) for c in cc]

    def _maximal_cliques(self):
        # find degeneracy ordering
        o = []
        p = set(self.v)
        while len(o) < len(self.v):
            v = min((len(set(self.ne[v]).difference(o)), v) for v in p)[1]
            o.append(v)
            p.remove(v)

        # brute-force bron_kerbosch algorithm
        c2 = set()

        def bron_kerbosch(r, p, x):
            if not p and not x:
                c2.add(tuple(sorted(r)))
            p = set(p)
            x = set(x)
            for v in list(p):
                bron_kerbosch(r.union({v}),
                              p.intersection(self.ne[v]),
                              x.intersection(self.ne[v]))
                p.remove(v)
                x.add(v)

        # apply brute-force bron_kerbosch with degeneracy ordering
        p = set(self.v)
        x = set()
        for v in o:
            bron_kerbosch({v},
                          p.intersection(self.ne[v]),
                          x.intersection(self.ne[v]))
            p.remove(v)
            x.add(v)

        return c2
    
    def identify(self, X, Y):
        """
        Takes sets of variables X and Y as input.
        If identifiable, returns P(Y | do(X)) in the form of a Pexpr object.
        Otherwise, returns FAIL.
        """
        Q_evals = dict()
        V_eval = Pexpr(upper=[Punit(self._set_v)], lower=[], marg_set=set())
        Q_evals[tuple(self.v)] = V_eval

        raw_C = self._set_v.difference(X)
        an_Y = self.subgraph(raw_C).ancestors(Y)
        marg = an_Y.difference(Y)

        Q_list = self.cc

        Qy_list = self.subgraph(an_Y).cc
        if len(Qy_list) == 1:
            Qy = set(Qy_list[0])
            for raw_Q in Q_list:
                Q = set(raw_Q)
                if Qy.issubset(Q):
                    self._evaluate_Q(Q, self._set_v, Q_evals)
                    result = self._identify_help(Qy, Q, Q_evals)
                    if result == "FAIL":
                        return "FAIL"
                    result.add_marg(marg)
                    return result
        else:
            upper = []
            for raw_Qy in Qy_list:
                Qy = set(raw_Qy)
                for raw_Q in Q_list:
                    Q = set(raw_Q)
                    if Qy.issubset(Q):
                        self._evaluate_Q(Q, self._set_v, Q_evals)
                        result = self._identify_help(Qy, Q, Q_evals)
                        if result == "FAIL":
                            return "FAIL"
                        upper.append(result)

            result = Pexpr(upper=upper, lower=[], marg_set=set())
            result.add_marg(marg)
            return result

    def _identify_help(self, C, T, Q_evals):
        T_eval = Q_evals[self._serialize(T)]
        if C == T:
            return T_eval

        an_C = self.subgraph(T).ancestors(C)
        if an_C == T:
            return "FAIL"

        marg_out = T.difference(an_C)
        an_C_eval = deepcopy(T_eval)
        an_C_eval.add_marg(marg_out)
        Q_evals[self._serialize(an_C)] = an_C_eval

        Q_list = self.subgraph(an_C).cc
        for raw_Q in Q_list:
            Q = set(raw_Q)
            if C.issubset(Q):
                self._evaluate_Q(Q, an_C, Q_evals)
                return self._identify_help(C, Q, Q_evals)

    def ancestors(self, C):
        """
        Returns the ancestors of set C.
        """
        assert C.issubset(self._set_v)

        frontier = [c for c in C]
        an = {c for c in C}
        while len(frontier) > 0:
            cur_v = frontier.pop(0)
            for par_v in self.pa[cur_v]:
                if par_v not in an:
                    an.add(par_v)
                    frontier.append(par_v)

        return an

    def _convert_set_to_sorted(self, C):
        return [v for v in self.v if v in C]

    def _serialize(self, C):
        return tuple(self._convert_set_to_sorted(C))

    def _evaluate_Q(self, A, B, Q_evals):
        """
        Given variable sets B and its subset A, with Q[B] stored in Q_evals, Q[A] is computed using Q[B] and
        stored in Q_evals.
        """
        assert A.issubset(B)
        assert B.issubset(self._set_v)

        A_key = self._serialize(A)
        if A_key in Q_evals:
            return

        A_list = self._convert_set_to_sorted(A)
        B_list = self._convert_set_to_sorted(B)
        B_eval = Q_evals[self._serialize(B)]

        upper = []
        lower = []

        start = 0
        i = 0
        j = 0
        while i < len(A_list):
            while A_list[i] != B_list[j]:
                j += 1
                start += 1

            while i < len(A_list) and A_list[i] == B_list[j]:
                i += 1
                j += 1

            up_term = deepcopy(B_eval)
            if j < len(B_list):
                up_term.add_marg(set(B_list[j:]))
            upper.append(up_term)
            if start != 0:
                low_term = deepcopy(B_eval)
                low_term.add_marg(set(B_list[start:]))
                lower.append(low_term)
            start = j

        Q_evals[A_key] = Pexpr(upper=upper, lower=lower, marg_set=set())

    @classmethod
    def read(cls, filename):
        with open(filename) as file:
            mode = None
            V = []
            directed_edges = []
            bidirected_edges = []
            try:
                for i, line in enumerate(map(str.strip, file), 1):
                    if line == '':
                        continue

                    m = re.match('<([A-Z]+)>', line)
                    if m:
                        mode = m.groups()[0]
                        continue

                    if mode == 'NODES':
                        if line.isidentifier():
                            V.append(line)
                        else:
                            raise ValueError('invalid identifier')
                    elif mode == 'EDGES':
                        if '<->' in line:
                            v1, v2 = map(str.strip, line.split('<->'))
                            bidirected_edges.append((v1, v2))
                        elif '->' in line:
                            v1, v2 = map(str.strip, line.split('->'))
                            directed_edges.append((v1, v2))
                        else:
                            raise ValueError('invalid edge type')
                    else:
                        raise ValueError('unknown mode')
            except Exception as e:
                raise ValueError(f'Error parsing line {i}: {e}: {line}')
            return cls(V, directed_edges, bidirected_edges)

    def save(self, filename):
        with open(filename, 'w') as file:
            lines = ["<NODES>\n"]
            for V in self.v:
                lines.append("{}\n".format(V))
            lines.append("\n")
            lines.append("<EDGES>\n")
            for V1, V2 in self.de:
                lines.append("{} -> {}\n".format(V1, V2))
            for V1, V2 in self.be:
                lines.append("{} <-> {}\n".format(V1, V2))
            file.writelines(lines)


def sample_cg(n, dir_rate, bidir_rate, enforce_direct_path=False, enforce_bidirect_path=False, enforce_ID=None):
    """
    Samples a random causal diagram with n variables, including X and Y.
    All directed edges are independently included with a chance of dir_rate.
    All bidirected edges are independently included with a chance of bidir_rate.
    enforce_direct_path: if True, then there is guaranteed to be a directed path from X to Y
        this implies almost surely that P(Y | do(X)) != P(Y)
    enforce_bidirect_path: if True, then there is guaranteed to be a bidirected path from X to Y
        this implies P(Y | do(X)) is not amenable to backdoor adjustment
    enforce_ID: if True, then P(Y | do(X)) is guaranteed to be identifiable
                if False, then P(Y | do(X)) is guaranteed to not be identifiable
    """
    cg = None
    done = False

    while not done:
        x_loc = random.randint(0, n - 2)
        V_list = ['V{}'.format(i + 1) for i in range(n - 2)]
        V_list.insert(x_loc, 'X')
        V_list.append('Y')

        de_list = []
        be_list = []
        for i in range(len(V_list) - 1):
            for j in range(i + 1, len(V_list)):
                if random.random() < dir_rate:
                    de_list.append((V_list[i], V_list[j]))
                if random.random() < bidir_rate:
                    be_list.append((V_list[i], V_list[j]))

        cg = CausalGraph(V_list, de_list, be_list)

        done = True
        if enforce_direct_path and not graph_search(cg, 'X', 'Y', edge_type="direct"):
            done = False
        if enforce_bidirect_path and not graph_search(cg, 'X', 'Y', edge_type="bidirect"):
            done = False

        if enforce_ID is not None:
            id_status = (cg.identify(X={'X'}, Y={'Y'}) != "FAIL")
            if enforce_ID != id_status:
                done = False

    return cg


def graph_search(cg, v1, v2, edge_type="direct"):
    """
    Uses BFS to check for a path between v1 and v2 in cg.
    """
    assert edge_type in ["direct", "bidirect"]
    assert v1 in cg._set_v
    assert v2 in cg._set_v

    q = deque([v1])
    seen = {v1}
    while len(q) > 0:
        cur = q.popleft()
        if edge_type == "direct":
            cur_ne = cg.ch[cur]
        else:
            cur_ne = cg.ne[cur]

        for ne in cur_ne:
            if ne not in seen:
                if ne == v2:
                    return True
                seen.add(ne)
                q.append(ne)

    return False


class Punit:
    def __init__(self, V_set):
        self.V_set = V_set

    def _marg_remove(self, V):
        if V in self.V_set:
            self.V_set.remove(V)

    def _marg_check_remove(self, V):
        if V in self.V_set:
            return True
        return False

    def _marg_check_contains(self, V):
        if V in self.V_set:
            return 1, 0
        return 0, 0

    def get_latex(self):
        return str(self)

    def __str__(self):
        out = "P("
        for V in self.V_set:
            out = out + V + ','
        out = out[:-1] + ')'
        return out


class Pexpr:
    def __init__(self, upper, lower, marg_set):
        self.upper = upper
        self.lower = lower
        self.marg_set = marg_set

    def add_marg(self, marg_V):
        for V in marg_V:
            if self._marg_check_remove(V):
                self._marg_remove(V)
            else:
                self.marg_set.add(V)

    def _marg_remove(self, V):
        for Pu in self.upper:
            Pu._marg_remove(V)

    def _marg_check_remove(self, V):
        upper_count, lower_count = self._marg_check_contains(V)
        if lower_count == 0 and upper_count <= 1:
            return True
        return False

    def _marg_check_contains(self, V):
        upper_count = 0
        lower_count = 0
        for Pu in self.upper:
            up, low = Pu._marg_check_contains(V)
            upper_count += up
            lower_count += low
        for Pl in self.lower:
            up, low = Pl._marg_check_contains(V)
            lower_count += up + low
        return upper_count, lower_count

    def get_latex(self):
        if len(self.marg_set) == 0 and len(self.lower) == 0 and len(self.upper) == 1:
            return self.upper[0].get_latex()

        out = "\\left["
        if len(self.marg_set) > 0:
            out = "\\sum_{"
            for M in self.marg_set:
                out = out + M + ','
            out = out[:-1] + '}\\left['

        if len(self.lower) > 0:
            out = out + "\\frac{"
            for P in self.upper:
                out = out + P.get_latex()
            out = out + "}{"
            for P in self.lower:
                out = out + P.get_latex()
            out = out + "}"
        else:
            for P in self.upper:
                out = out + P.get_latex()

        out = out + '\\right]'
        return out

    def __str__(self):
        if len(self.marg_set) == 0 and len(self.lower) == 0 and len(self.upper) == 1:
            return str(self.upper[0])

        out = "["
        if len(self.marg_set) > 0:
            out = "sum{"
            for M in self.marg_set:
                out = out + M + ','
            out = out[:-1] + '}['
        for P in self.upper:
            out = out + str(P)
        if len(self.lower) > 0:
            out = out + " / "
            for P in self.lower:
                out = out + str(P)
        out = out + ']'
        return out


if __name__ == "__main__":
    #cg = CausalGraph.read("../../dat/cg/napkin.cg")
    cg = sample_cg(10, 0.3, 0.2, enforce_direct_path=False, enforce_bidirect_path=False, enforce_ID=False)
    result = cg.identify(X={'X'}, Y={'Y'})
    print(result)
    if result != "FAIL":
        print(result.get_latex())
