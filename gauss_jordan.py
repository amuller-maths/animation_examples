from manim import *
from sympy import sympify


class Test(Scene):
    def construct(self):
        text1 = Tex("ABC", fill_opacity=1)
        text2 = Tex("ABC", fill_opacity=0.3).next_to(text1, DOWN)

        self.add(text1, text2)
        self.wait(1)


class GaussJordan(Scene):
    def construct(self):
        mat = [[1, 2, -1, 1, 2], [1, 0, 1, 3, 0], [2, 2, 1, 2, 1]]
        mat = [[sympify(m) for m in mm] for mm in mat]
        n, p = len(mat), len(mat[0])
        mob_mat = Matrix(
            mat,
            bracket_h_buff=MED_LARGE_BUFF,
            left_bracket="\\big(",
            right_bracket="\\big)",
            element_alignment_corner=ORIGIN,
        )
        self.add(mob_mat)
        rows = mob_mat.get_rows()

        indice_courant = 0
        for j in range(p):
            indice_pivot = self.cherche_pivot(
                mat=mat, rows=rows, current_col=j, current_row=indice_courant
            )
            if indice_pivot != -1:
                pivot = mat[indice_pivot][j]
                if indice_pivot != indice_courant:
                    self.echange_lignes(mat, rows, indice_courant, indice_pivot)
                for i in range(indice_courant + 1, n):
                    alpha = mat[i][j]
                    if alpha != 0:
                        self.transvection(
                            mat, rows, i, indice_courant, -alpha / pivot, j
                        )
                indice_courant += 1
            if indice_courant >= n:
                break

        self.play(rows.set_opacity, 1)
        self.add(Tex("C'est gagné !"))

    def echange_lignes(self, mat, rows, i, j):
        """Li <-> Lj"""
        text_string = (
            "On échange les lignes {i} et {j} : $L_{i} \leftrightarrow L_{j}$".format(
                i=i + 1, j=j + 1
            )
        )
        text = Tex(text_string).shift(UP + rows.get_top())
        mat[i], mat[j] = mat[j], mat[i]

        A, D = rows[i].get_left() + LEFT, rows[j].get_left() + LEFT
        D[0] = A[0]
        B, C = A + LEFT, D + LEFT
        double_arrow = VGroup(Arrow(B, A, buff=0), Line(B, C), Arrow(C, D, buff=0))
        self.play(FadeIn(text), FadeIn(double_arrow))
        self.wait(1)
        self.play(*[Swap(rows[i][k], rows[j][k]) for k in range(len(rows[i]))])

        rows[i], rows[j] = rows[j], rows[i]

        self.remove(double_arrow)
        self.remove(text)
        self.wait(2)

    def transvection(self, mat, rows, i, j, alpha, current_col):
        """Li <- Li + alpha Lj"""
        n, p = len(mat), len(mat[0])
        if alpha == 1:
            template = "$L_{i} \leftarrow L_{i} + L_{j}$"
        elif alpha == -1:
            template = "$L_{i} \leftarrow L_{i} - L_{j}$"
        elif alpha > 0:
            template = "$L_{i} \leftarrow L_{i} + {alpha}L_{j}$"
        else:
            template = "$L_{i} \leftarrow L_{i} {alpha}L_{j}$"

        text_string = "On effectue l'opération " + template.format(
            i=i + 1, j=j + 1, alpha=alpha
        )

        text = Tex(text_string).shift(UP + rows.get_top())

        mat[i] = [mat[i][k] + alpha * mat[j][k] for k in range(len(mat[0]))]

        new_mob_mat = Matrix(
            mat,
            bracket_h_buff=MED_LARGE_BUFF,
            left_bracket="\\big(",
            right_bracket="\\big)",
            element_alignment_corner=ORIGIN,
        )
        new_rows = new_mob_mat.get_rows()
        # keeping rows well aligned
        for index_col in range(current_col, p):
            new_rows[i][index_col].move_to(rows[i][index_col].get_center())
        A, D = rows[i].get_left() + LEFT, rows[j].get_left() + LEFT
        D[0] = A[0]
        B, C = A + LEFT, D + LEFT
        simple_arrow = VGroup(
            Line(D, C, buff=0), Line(C, B, buff=0), Arrow(B, A, buff=0)
        )
        text_arrow = "$\\times$ {alpha}" if alpha > 0 else "$\\times$ ({alpha})"
        text_arrow = Tex(text_arrow.format(alpha=alpha)).next_to(simple_arrow[1], LEFT)

        self.play(FadeIn(text))
        self.play(ShowCreation(simple_arrow), FadeIn(text_arrow))
        self.wait(1)
        self.play(Transform(rows[i][current_col:], new_rows[i][current_col:]))

        self.wait(2)
        self.remove(simple_arrow)

        self.remove(text)
        self.remove(text_arrow)

    def cherche_pivot(self, mat, rows, current_col, current_row):
        text_string = "On cherche un pivot dans la colonne {col}".format(
            col=current_col + 1
        )
        text = Tex(text_string).shift(UP + rows[0].get_top())
        self.play(FadeIn(text))
        circle = Circle(
            radius=rows[current_row][current_col].get_height(),
            color=RED,
            arc_center=rows[current_row][current_col].get_center(),
        )
        self.play(FadeIn(circle))
        for i in range(current_row, len(rows)):

            if mat[i][current_col] != 0:
                new_circle = Circle(
                    radius=rows[i][current_col].get_height(),
                    color=YELLOW,
                    arc_center=rows[i][current_col].get_center(),
                )
                self.play(Transform(circle, new_circle))
                rows[i][current_col].submobjects.append(circle)
                self.wait(1)
                self.remove(text)
                return i
            else:
                new_circle = Circle(
                    radius=rows[i][current_col].get_height(),
                    color=RED,
                    arc_center=rows[i][current_col].get_center(),
                )
                self.play(Transform(circle, new_circle))
                self.wait(1)
                self.remove(circle)
        self.remove(text)
        self.wait(1)
        return -1


class GaussJordanUnique(Scene):
    def construct(self):
        mat = [[1, 0, 2, 1, 4], [3, 2, 10, 5, 4], [2, -1, 2, 2, 13], [3, -1, 4, 4, 18]]
        mat = [[Fraction(m) for m in mm] for mm in mat]
        n, p = len(mat), len(mat[0])
        mob_mat = Matrix(
            mat,
            bracket_h_buff=MED_LARGE_BUFF,
            left_bracket="\\big(",
            right_bracket="\\big)",
            element_alignment_corner=ORIGIN,
        )
        self.add(mob_mat)
        rows = mob_mat.get_rows()

        indice_courant = 0
        for j in range(p):
            indice_pivot = self.cherche_pivot(
                mat=mat, rows=rows, current_col=j, current_row=indice_courant
            )
            if indice_pivot != -1:
                pivot = mat[indice_pivot][j]
                if indice_pivot != indice_courant:
                    self.echange_lignes(mat, rows, indice_courant, indice_pivot)
                if pivot != 1:
                    self.dilatation(mat, rows, indice_courant, 1 / pivot, j)

                for i in range(n):
                    if i != indice_courant:
                        alpha = mat[i][j]
                        if alpha != 0:
                            self.transvection(mat, rows, i, indice_courant, -alpha, j)
                indice_courant += 1
            if indice_courant >= n:
                break
        text = "C'est la forme échelonnée réduite."
        text = Tex(text).shift(UP + rows.get_top())
        self.add(text)
        self.wait(3)

    def echange_lignes(self, mat, rows, i, j):
        """Li <-> Lj"""
        text_string = (
            "On échange les lignes {i} et {j} : $L_{i} \leftrightarrow L_{j}$".format(
                i=i + 1, j=j + 1
            )
        )
        text = Tex(text_string).shift(UP + rows.get_top())
        mat[i], mat[j] = mat[j], mat[i]

        A, D = rows[i].get_left() + LEFT, rows[j].get_left() + LEFT
        D[0] = A[0]
        B, C = A + LEFT, D + LEFT
        double_arrow = VGroup(Arrow(B, A, buff=0), Line(B, C), Arrow(C, D, buff=0))
        self.play(FadeIn(text), FadeIn(double_arrow))
        self.wait(1)
        self.play(*[Swap(rows[i][k], rows[j][k]) for k in range(len(rows[i]))])

        rows[i], rows[j] = rows[j], rows[i]

        self.remove(double_arrow)
        self.remove(text)
        self.wait(2)

    def transvection(self, mat, rows, i, j, alpha, current_col):
        """Li <- Li + alpha Lj"""
        n, p = len(mat), len(mat[0])
        if alpha == 1:
            template = "$L_{i} \leftarrow L_{i} + L_{j}$"
        elif alpha == -1:
            template = "$L_{i} \leftarrow L_{i} - L_{j}$"
        elif alpha > 0:
            template = "$L_{i} \leftarrow L_{i} + {alpha}L_{j}$"
        else:
            template = "$L_{i} \leftarrow L_{i} {alpha}L_{j}$"

        text_string = "On effectue l'opération " + template.format(
            i=i + 1, j=j + 1, alpha=alpha
        )

        text = Tex(text_string).shift(UP + rows.get_top())

        mat[i] = [mat[i][k] + alpha * mat[j][k] for k in range(len(mat[0]))]

        new_mob_mat = Matrix(
            mat,
            bracket_h_buff=MED_LARGE_BUFF,
            left_bracket="\\big(",
            right_bracket="\\big)",
            element_alignment_corner=ORIGIN,
        )
        new_rows = new_mob_mat.get_rows()
        # keeping rows well aligned
        for index_col in range(current_col, p):
            new_rows[i][index_col].move_to(rows[i][index_col].get_center())
        A, D = rows[i].get_left() + LEFT, rows[j].get_left() + LEFT
        D[0] = A[0]
        B, C = A + LEFT, D + LEFT
        simple_arrow = VGroup(
            Line(D, C, buff=0), Line(C, B, buff=0), Arrow(B, A, buff=0)
        )
        text_arrow = "$\\times$ {alpha}" if alpha > 0 else "$\\times$ ({alpha})"
        text_arrow = Tex(text_arrow.format(alpha=alpha)).next_to(simple_arrow[1], LEFT)

        self.play(FadeIn(text))
        self.play(ShowCreation(simple_arrow), FadeIn(text_arrow))
        self.wait(1)
        self.play(Transform(rows[i][current_col:], new_rows[i][current_col:]))

        self.wait(2)
        self.remove(simple_arrow)

        self.remove(text)
        self.remove(text_arrow)

    def dilatation(self, mat, rows, i, alpha, current_col):
        n, p = len(mat), len(mat[0])
        text_string = "On multiplie la ligne {i} par {alpha} : $L_{i} \leftarrow {alpha} L_{i}$".format(
            i=i + 1, alpha=alpha
        )
        text = Tex(text_string).shift(UP + rows.get_top())

        mat[i] = [alpha * mat[i][k] for k in range(p)]
        new_mob_mat = Matrix(
            mat,
            bracket_h_buff=MED_LARGE_BUFF,
            left_bracket="\\big(",
            right_bracket="\\big)",
            element_alignment_corner=ORIGIN,
        )
        new_rows = new_mob_mat.get_rows()
        # keeping rows well aligned
        for index_col in range(current_col, p):
            new_rows[i][index_col].move_to(rows[i][index_col].get_center())

            for submob in rows[i][index_col].submobjects:
                if isinstance(submob, Circle):
                    circle = Circle(
                        radius=new_rows[i][index_col].get_height(),
                        color=YELLOW,
                        arc_center=rows[i][index_col].get_center(),
                    )
                    new_rows[i][index_col].submobjects.append(circle)

        A = rows[i].get_left() + LEFT
        B = A + LEFT
        arrow = Arrow(B, A, buff=0)
        text_arrow = "$\\times$ {alpha}" if alpha > 0 else "$\\times$ ({alpha})"
        text_arrow = Tex(text_arrow.format(alpha=alpha)).next_to(arrow, LEFT)

        self.play(FadeIn(text))
        self.play(ShowCreation(arrow), FadeIn(text_arrow))
        self.wait(1)
        self.play(Transform(rows[i][current_col:], new_rows[i][current_col:]))

        self.wait(2)
        self.remove(arrow)

        self.remove(text)
        self.remove(text_arrow)

    def cherche_pivot(self, mat, rows, current_col, current_row):
        text_string = "On cherche un pivot dans la colonne {col}".format(
            col=current_col + 1
        )
        text = Tex(text_string).shift(UP + rows[0].get_top())
        self.play(FadeIn(text))
        circle = Circle(
            radius=rows[current_row][current_col].get_height(),
            color=RED,
            arc_center=rows[current_row][current_col].get_center(),
        )
        self.play(FadeIn(circle))
        for i in range(current_row, len(rows)):

            if mat[i][current_col] != 0:
                new_circle = Circle(
                    radius=rows[i][current_col].get_height(),
                    color=YELLOW,
                    arc_center=rows[i][current_col].get_center(),
                )
                self.play(Transform(circle, new_circle))
                rows[i][current_col].submobjects.append(circle)
                self.wait(1)
                self.remove(text)
                return i
            else:
                new_circle = Circle(
                    radius=rows[i][current_col].get_height(),
                    color=RED,
                    arc_center=rows[i][current_col].get_center(),
                )
                self.play(Transform(circle, new_circle))
                self.wait(1)
                self.remove(circle)
        self.remove(text)
        self.wait(1)
        return -1


class System(Scene):
    def construct(self):
        mat = [[1, -1, 1, 1, 0], [3, -3, 3, 2, 0], [1, -1, 1, 0, 0], [5, -5, 5, 7, 0]]
        mat = [[sympify(m) for m in mm] for mm in mat]
        n, p = len(mat), len(mat[0])

        mob_mat = Matrix(
            mat,
            bracket_h_buff=MED_LARGE_BUFF,
            left_bracket="\\big(",
            right_bracket="\\big)",
            element_alignment_corner=ORIGIN,
        )
        unknowns = {
            1: ["x", "y", "z", "t"],
            2: ["a", "b", "c", "d", "e", "f", "g", "h"],
            3: ["x_{{{}}}".format(i) for i in range(1, p + 1)],
        }

        system = self.get_system_from_mat(mat, unknowns[1])

        text_intro = Tex("L'objectif est de résoudre le système suivant :")
        text_intro.next_to(system, UP)
        text_expl1 = Tex("Résolvons ce système par la méthode du pivot").next_to(
            system, DOWN
        )
        text_expl2 = Tex(
            "Pour cela, raisonnons sur la matrice augmentée du système"
        ).next_to(text_expl1, DOWN)

        intro_group = VGroup(text_intro, system, text_expl1, text_expl2)

        self.play(FadeIn(intro_group[0]), FadeIn(intro_group[1]))
        self.wait(2)
        self.play(FadeIn(intro_group[2], intro_group[3]))
        self.wait(3)
        self.play(FadeOut(intro_group))
        self.play(FadeIn(mob_mat))

        rows = mob_mat.get_rows()

        indice_courant = 0
        for j in range(p):
            indice_pivot = self.cherche_pivot(
                mat=mat, rows=rows, current_col=j, current_row=indice_courant
            )
            if indice_pivot != -1:
                pivot = mat[indice_pivot][j]
                if indice_pivot != indice_courant:
                    self.echange_lignes(mat, rows, indice_courant, indice_pivot)
                if pivot != 1:
                    self.dilatation(mat, rows, indice_courant, 1 / pivot, j)

                for i in range(n):
                    if i != indice_courant:
                        alpha = mat[i][j]
                        if alpha != 0:
                            self.transvection(mat, rows, i, indice_courant, -alpha, j)
                indice_courant += 1
            if indice_courant >= n:
                break
        text = "C'est la forme échelonnée réduite."
        text = Tex(text).shift(UP + rows.get_top())
        self.add(text)
        self.wait(3)
        self.play(FadeOut(text), FadeOut(mob_mat))

        system = self.get_system_from_mat(mat, unknowns[1])

        text_intro = Tex("Le système de départ est équivalent au système suivant :")
        text_intro.next_to(system, UP)
        self.play(FadeIn(text_intro))
        self.play(FadeIn(system))
        self.wait(3)

        text_solution = self.get_solutions(mat, unknowns[1])
        text_solution.next_to(system, DOWN)
        self.play(FadeIn(text_solution))
        self.wait(3)

    def echange_lignes(self, mat, rows, i, j):
        """Li <-> Lj"""
        text_string = (
            "On échange les lignes {i} et {j} : $L_{i} \leftrightarrow L_{j}$".format(
                i=i + 1, j=j + 1
            )
        )
        text = Tex(text_string).shift(UP + rows.get_top())
        mat[i], mat[j] = mat[j], mat[i]

        A, D = rows[i].get_left() + LEFT, rows[j].get_left() + LEFT
        D[0] = A[0]
        B, C = A + LEFT, D + LEFT
        double_arrow = VGroup(Arrow(B, A, buff=0), Line(B, C), Arrow(C, D, buff=0))
        self.play(FadeIn(text), FadeIn(double_arrow))
        self.wait(1)
        self.play(*[Swap(rows[i][k], rows[j][k]) for k in range(len(rows[i]))])

        rows[i], rows[j] = rows[j], rows[i]

        self.remove(double_arrow)
        self.remove(text)
        self.wait(2)

        system_rows = [
            [self.coeff_system(mat[i][j], unknowns[1][j], j) for j in range(p - 1)]
            for i in range(n)
        ]

        system_string = "".join(
            [self.row_system(system_rows[i], mat[i][-1], n - 1 - i) for i in range(n)]
        )

    def transvection(self, mat, rows, i, j, alpha, current_col):
        """Li <- Li + alpha Lj"""
        n, p = len(mat), len(mat[0])
        try:
            if alpha == 1:
                template = "$L_{i} \leftarrow L_{i} + L_{j}$"
            elif alpha == -1:
                template = "$L_{i} \leftarrow L_{i} - L_{j}$"
            elif alpha > 0:
                template = "$L_{i} \leftarrow L_{i} + {alpha}L_{j}$"
            else:
                template = "$L_{i} \leftarrow L_{i} {alpha}L_{j}$"
        except TypeError:
            template = "$L_{i} \leftarrow L_{i} + ({alpha})L_{j}$"

        text_string = "On effectue l'opération " + template.format(
            i=i + 1, j=j + 1, alpha=alpha
        )

        text = Tex(text_string).shift(UP + rows.get_top())

        mat[i] = [mat[i][k] + alpha * mat[j][k] for k in range(len(mat[0]))]

        new_mob_mat = Matrix(
            mat,
            bracket_h_buff=MED_LARGE_BUFF,
            left_bracket="\\big(",
            right_bracket="\\big)",
            element_alignment_corner=ORIGIN,
        )
        new_rows = new_mob_mat.get_rows()
        # keeping rows well aligned
        for index_col in range(current_col, p):
            new_rows[i][index_col].move_to(rows[i][index_col].get_center())
        A, D = rows[i].get_left() + LEFT, rows[j].get_left() + LEFT
        D[0] = A[0]
        B, C = A + LEFT, D + LEFT
        simple_arrow = VGroup(
            Line(D, C, buff=0), Line(C, B, buff=0), Arrow(B, A, buff=0)
        )
        try:
            text_arrow = "$\\times$ {alpha}" if alpha > 0 else "$\\times$ ({alpha})"
        except TypeError:
            text_arrow = "$\\times$ ({alpha})"
        text_arrow = Tex(text_arrow.format(alpha=alpha)).next_to(simple_arrow[1], LEFT)

        self.play(FadeIn(text))
        self.play(ShowCreation(simple_arrow), FadeIn(text_arrow))
        self.wait(1)
        self.play(Transform(rows[i][current_col:], new_rows[i][current_col:]))

        self.wait(2)
        self.remove(simple_arrow)

        self.remove(text)
        self.remove(text_arrow)

    def dilatation(self, mat, rows, i, alpha, current_col):
        n, p = len(mat), len(mat[0])
        text_string = "On multiplie la ligne {i} par {alpha} : $L_{i} \leftarrow {alpha} L_{i}$".format(
            i=i + 1, alpha=alpha
        )
        text = Tex(text_string).shift(UP + rows.get_top())

        mat[i] = [alpha * mat[i][k] for k in range(p)]
        new_mob_mat = Matrix(
            mat,
            bracket_h_buff=MED_LARGE_BUFF,
            left_bracket="\\big(",
            right_bracket="\\big)",
            element_alignment_corner=ORIGIN,
        )
        new_rows = new_mob_mat.get_rows()
        # keeping rows well aligned
        for index_col in range(current_col, p):
            new_rows[i][index_col].move_to(rows[i][index_col].get_center())

            for submob in rows[i][index_col].submobjects:
                if isinstance(submob, Circle):
                    circle = Circle(
                        radius=new_rows[i][index_col].get_height(),
                        color=YELLOW,
                        arc_center=rows[i][index_col].get_center(),
                    )
                    new_rows[i][index_col].submobjects.append(circle)

        A = rows[i].get_left() + LEFT
        B = A + LEFT
        arrow = Arrow(B, A, buff=0)
        try:
            text_arrow = "$\\times$ {alpha}" if alpha > 0 else "$\\times$ ({alpha})"
        except TypeError:
            text_arrow = "$\\times$ ({alpha})"
        text_arrow = Tex(text_arrow.format(alpha=alpha)).next_to(arrow, LEFT)

        self.play(FadeIn(text))
        self.play(ShowCreation(arrow), FadeIn(text_arrow))
        self.wait(1)
        self.play(Transform(rows[i][current_col:], new_rows[i][current_col:]))

        self.wait(2)
        self.remove(arrow)

        self.remove(text)
        self.remove(text_arrow)

    def cherche_pivot(self, mat, rows, current_col, current_row):
        text_string = "On cherche un pivot dans la colonne {col}".format(
            col=current_col + 1
        )
        text = Tex(text_string).shift(UP + rows[0].get_top())
        self.play(FadeIn(text))
        circle = Circle(
            radius=rows[current_row][current_col].get_height(),
            color=RED,
            arc_center=rows[current_row][current_col].get_center(),
        )
        self.play(FadeIn(circle))
        for i in range(current_row, len(rows)):

            if mat[i][current_col] != 0:
                new_circle = Circle(
                    radius=rows[i][current_col].get_height(),
                    color=YELLOW,
                    arc_center=rows[i][current_col].get_center(),
                )
                self.play(Transform(circle, new_circle))
                rows[i][current_col].submobjects.append(circle)
                self.wait(1)
                self.remove(text)
                return i
            else:
                new_circle = Circle(
                    radius=rows[i][current_col].get_height(),
                    color=RED,
                    arc_center=rows[i][current_col].get_center(),
                )
                self.play(Transform(circle, new_circle))
                self.wait(1)
                self.remove(circle)
        self.remove(text)
        self.wait(1)
        return -1

    def coeff_system(self, coeff, ukn, ind=0):
        if ind == 0:
            prefix = ""
        elif coeff > 0:
            prefix = "+"
        else:
            prefix = ""
        if coeff == 0:
            return ""
        elif coeff == -1:
            return "-" + ukn
        elif coeff == 1:
            return prefix + ukn
        else:
            return prefix + str(coeff) + ukn

    def sec_mem(self, coeff, ind=1):
        if coeff == 0 and ind != 1:
            return ""
        else:
            return str(coeff)

    def row_system(self, expr_row, sec_mem, ind=0):
        suffix = "" if ind == 0 else " \\\\\n"
        return "".join(expr_row) + "=" + str(sec_mem) + suffix

    def get_system_from_mat(self, mat, unknowns):
        n, p = len(mat), len(mat[0])
        system_rows = []
        for i in range(n):
            if mat[i] != [0 for j in range(p)]:
                ind = 0
                L = []
                for j in range(p - 1):
                    L.append(self.coeff_system(mat[i][j], unknowns[j], ind))
                    if mat[i][j] != 0:
                        ind = 1
                system_rows.append(L)
        system_string = "".join(
            [
                self.row_system(system_rows[i], mat[i][-1], n - 1 - i)
                for i in range(len(system_rows))
            ]
        )
        prefix = "\\begin{cases} "
        suffix = " \\end{cases}"
        system_string = prefix + system_string + suffix
        system = MathTex(system_string)

        return system

    def get_solutions(self, mat, unknowns):
        n, p = len(mat), len(mat[0])
        liste_solutions = unknowns[: p - 1].copy()
        liste_rangs_pivots = []
        for i in range(n):
            pivot = next((j for j, x in enumerate(mat[i]) if x), None)
            if pivot is not None:
                s = self.sec_mem(mat[i][-1], sum([abs(m) for m in mat[i][:-1]]))
                ind = 0
                for j in range(pivot + 1, p - 1):
                    s += self.coeff_system(-mat[i][j], unknowns[j], ind)
                    if mat[i][j] != 0:
                        ind = 1
                liste_rangs_pivots.append(pivot)
                liste_solutions[pivot] = s

        parametres = [
            unknowns[i] for i in range(len(unknowns) - 1) if i not in liste_rangs_pivots
        ]

        if len(parametres) > 1:
            phrase_intro = Tex("L'ensemble des solutions du système est donc :")
            sols = Tex(
                "$\\left\\{ ("
                + ",".join(liste_solutions)
                + "), ("
                + ",".join(parametres)
                + ") \\in \\mathbb{{R}}^{num_param}".format(
                    param=str(tuple(parametres)),
                    num_param=len(parametres),
                )
                + " \\right\\}$"
            ).next_to(phrase_intro, DOWN)
            return VGroup(phrase_intro, sols)

        elif len(parametres) == 1:
            phrase_intro = Tex("L'ensemble des solutions du système est donc :")
            sols = Tex(
                "$\\left\\{ ("
                + ",".join(liste_solutions)
                + "), "
                + parametres[0]
                + " \\in \\mathbb{{R}}".format(
                    param=str(tuple(parametres)),
                    num_param=len(parametres),
                )
                + " \\right\\}$"
            ).next_to(phrase_intro, DOWN)
            return VGroup(phrase_intro, sols)
        else:
            return Tex(
                "L'unique solution du système est $({})$".format(
                    ",".join([str(mat[i][-1]) for i in range(n)])
                )
            )