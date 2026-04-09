import marimo

__generated_with = "0.23.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell
def _(np):
    A = np.array([[1,2,3],
                  [4,5,6]])
    A
    return (A,)


@app.cell
def _(A, np):
    x = np.array([1,2,3])
    # so this treats the vector as a column vector
    A@x
    return


@app.cell
def _(A, np):
    y = np.array([1,2])
    # now this gets treated as a row vector
    y@A
    return


@app.cell
def _(A, np):
    # but now if we have a list of vectors instead...
    X = np.array([[1,2,3],
                 [3,2,1]])

    # now we need to transpose
    A@X.transpose()
    return


@app.cell
def _(A, np):
    # However, if we use row vectors instead
    Y = np.array([[1, 2],
                 [2, 1]])
    # works just fine..., no wonder we use it.
    Y @ A
    return


@app.cell
def _(np):
    # Now let's figure out how to do Hadamard products
    a = np.array([1,2,3])
    b = np.array([3,2,1])
    # they're commutative so we don't even need to worry about stuff
    a * b
    return a, b


@app.cell(hide_code=True)
def _(a, b, np):
    # now we need to learn about outer products
    np.outer(a,b)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This should be something like

    $$
    \vec{a} \otimes \vec{b} = \begin{bmatrix}
    a_1\\a_2\\\vdots\\ a_n\\
    \end{bmatrix}\begin{bmatrix}
    b_1&b_2&\cdots&b_n\\
    \end{bmatrix}
    $$

    But everything in numpy is a row vector.
    """)
    return


@app.cell
def _(a, b, np):
    np.outer(b,a)
    return


@app.cell
def _(np):
    # sampling from a normal distribution.
    np.random.normal(3, 2, size=(3,5))
    return


if __name__ == "__main__":
    app.run()
