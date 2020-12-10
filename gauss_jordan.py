from manim import *
from fractions import Fraction


class Gauss_Jordan(Scene):
    def construct(self):
        mat = [[1, 1, 3, -1], [0, 2, -4, 5], [0, 1, 0, 3]]
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

        indice_courant = 0
        for j in range(p):
            indice_pivot = self.cherche_pivot(
                mat=mat, mob_mat=mob_mat, col=j, row=indice_courant
            )
            if indice_pivot != -1:
                pivot = mat[indice_pivot][j]
                if indice_pivot != indice_courant:
                    mob_mat = self.echange_lignes(
                        mat, mob_mat, indice_courant, indice_pivot
                    )
                for i in range(indice_courant + 1, n):

                    alpha = mat[i][j]
                    if alpha != 0:
                        mob_mat = self.transvection(
                            mat, mob_mat, i, indice_courant, -alpha / pivot
                        )
                indice_courant += 1
                if indice_courant >= n:
                    break
        self.add(Tex("C'est gagné !"))

    def echange_lignes(self, mat, mob_mat, i, j):
        """Li <-> Lj"""
        text_string = (
            "On échange les lignes {i} et {j} : $L_{i} \leftrightarrow L_{j}$".format(
                i=i + 1, j=j + 1
            )
        )
        text = Tex(text_string).shift(UP + mob_mat.get_top())
        mat[i], mat[j] = mat[j], mat[i]
        rows = mob_mat.get_rows()
        new_mob_mat = Matrix(
            mat,
            bracket_h_buff=MED_LARGE_BUFF,
            left_bracket="\\big(",
            right_bracket="\\big)",
        )
        A, D = rows[i].get_left() + LEFT, rows[j].get_left() + LEFT
        D[0] = A[0]
        B, C = A + LEFT, D + LEFT
        double_arrow = VGroup(Arrow(B, A, buff=0), Line(B, C), Arrow(C, D, buff=0))
        self.play(FadeIn(text), FadeIn(double_arrow))
        self.wait(1)
        self.play(Swap(rows[i], rows[j]))
        self.remove(mob_mat[1])
        self.remove(mob_mat[2])
        for sub in mob_mat[0]:
            self.remove(sub)
        self.remove(double_arrow)
        self.remove(text)

        self.add(new_mob_mat)
        self.wait(2)
        return new_mob_mat

    def transvection(self, mat, mob_mat, i, j, alpha):
        """Li <- Li + alpha Lj"""

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

        text = Tex(text_string).shift(UP + mob_mat.get_top())

        mat[i] = [mat[i][k] + alpha * mat[j][k] for k in range(len(mat[0]))]

        new_mob_mat = Matrix(
            mat,
            bracket_h_buff=MED_LARGE_BUFF,
            left_bracket="\\big(",
            right_bracket="\\big)",
            element_alignment_corner=ORIGIN,
        )
        rows = mob_mat.get_rows()
        new_rows = new_mob_mat.get_rows()
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
        # self.play(Transform(rows[i], new_rows[i]))
        self.play(Transform(rows, new_rows))
        self.wait(2)
        self.remove(simple_arrow)
        self.remove(mob_mat[1])
        self.remove(mob_mat[2])
        for sub in mob_mat[0]:
            self.remove(sub)
        self.remove(text)
        self.remove(text_arrow)
        self.add(new_mob_mat)
        return new_mob_mat

    def cherche_pivot(self, mat, mob_mat, col, row):
        print("col = ", col)
        print("row = ", row)
        text_string = "On cherche un pivot dans la colonne {col}".format(col=col + 1)
        text = Tex(text_string).shift(UP + mob_mat.get_top())
        self.play(FadeIn(text))
        cols = mob_mat.get_columns()
        circle = Circle(
            radius=cols[col][row].get_height(),
            color=RED,
            arc_center=cols[col][row].get_center(),
        )
        self.play(FadeIn(circle))
        for i in range(row, len(cols[0])):

            if mat[i][col] != 0:
                new_circle = Circle(
                    radius=cols[col][i].get_height(),
                    color=YELLOW,
                    arc_center=cols[col][i].get_center(),
                )
                self.play(Transform(circle, new_circle))
                cols[col][i].submobjects.append(circle)
                self.wait(1)
                self.remove(text)
                return i
            else:
                new_circle = Circle(
                    radius=cols[col][i].get_height(),
                    color=RED,
                    arc_center=cols[col][i].get_center(),
                )
                self.play(Transform(circle, new_circle))
                self.wait(1)
        self.remove(text)
        self.wait(1)
        return -1
