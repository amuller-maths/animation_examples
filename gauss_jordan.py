from manim import *
from fractions import Fraction


class Test(Scene):
    def construct(self):
        text1 = Tex("ABC", fill_opacity=1)
        text2 = Tex("ABC", fill_opacity=0.3).next_to(text1, DOWN)

        self.add(text1, text2)
        self.wait(1)


class GaussJordan(Scene):
    def construct(self):
        mat = [[0, 1, 4, 10], [0, 0, 0, 0], [2, 2, 4, 12], [4, 2, 1, 7]]
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
        mat = [[0, 1, 4, 10], [0, 0, 0, 0], [2, 2, 4, 12], [4, 2, 1, 7]]
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
                for i in range(n):
                    if i != indice_courant:
                        alpha = mat[i][j]
                        if alpha != 0:
                            self.transvection(
                                mat, rows, i, indice_courant, -alpha / pivot, j
                            )
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
        pass

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
