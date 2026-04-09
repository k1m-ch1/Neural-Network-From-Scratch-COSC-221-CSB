import marimo

__generated_with = "0.23.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Outer product

    Compute the outer product of

    ```python
    a = np.array([1, 2, 3])
    b = np.array([4, 5])
    ```

    Expected result:

    ```
    [[ 4  5]
    [ 8 10]
    [12 15]]
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The math

    Basically, we have:

    $$
    a = \begin{bmatrix}
    1\\
    2\\
    3\\
    \end{bmatrix}
    $$

    and
    $$
    b = \begin{bmatrix}
    4\\
    5\\
    \end{bmatrix}
    $$

    To compute the outer product, we do:

    $$
    ab^T = \begin{bmatrix}
    1\\
    2\\
    3\\
    \end{bmatrix}\begin{bmatrix}4&5\end{bmatrix}
    = \begin{bmatrix}
    4&5\\
    8&10\\
    12&15\\
    \end{bmatrix}
    $$
    """)
    return


@app.cell
def _(np):
    def _():
        a = np.array([1, 2, 3])
        b = np.array([4, 5])
        return a[:, None]@b[None,:]
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    compute the outer product but reverse the order
    """)
    return


@app.cell
def _(np):
    def _():
        a = np.array([1, 2, 3])
        b = np.array([4, 5])
        return b[:, None]@a[None,:]
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Batch outer products

    ```python
    A = np.array([[1, 2],
                  [3, 4]])

    B = np.array([[10, 20],
                  [30, 40]])
    ```
    """)
    return


@app.cell
def _(np):
    def _():
        A = np.array([[1, 2],
                      [3, 4]])

        B = np.array([[10, 20],
                      [30, 40]])
        print(A[:,:,None])
        print(B[:, None, :])
        return np.sum(A[:,:,None] @ B[:, None, :], axis=0)

    _()
    return


@app.cell
def _(np):
    def _():
        a = np.array([1, 2, 3])
        b = np.array([10, 20])
        return a[:,None]@b[None,:]

    _()
    return


@app.cell
def _(np):
    def _():
        Z = np.array([[1, 2, 3],
                      [4, 5, 6]])
        b = np.array([10, 20, 30])
        # broadcast the bias...
        return Z + b

    _()
    return


@app.cell
def _(np):
    def _():
        Z = np.array([[1, 2, 3],
                      [4, 5, 6]])
        b = np.array([10, 20])
        print(Z)
        print(b[:,None])
        return Z + b[:,None]
    _()
    return


@app.cell
def _(np):
    def _():
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])

        print(f"Dot:\n{(a[None,:]@b[:,None])[0][0]}")
        print(f"Outer:\n{a[:,None]@b[None,:]}")
    _()
    return


@app.cell
def _(np):
    def _():
        X = np.array([[1, 2],
                      [3, 4]])
        W = np.array([[10, 20],
                      [30, 40]])
        return X @ W
    _()
    return


@app.cell
def _(np):
    def _():
        A_prev = np.array([[1, 2],
                           [3, 4]])
        delta = np.array([[10, 20],
                          [30, 40]])

        dW = A_prev.T @ delta
        return dW

    _()
    return


@app.cell
def _(np):
    def _():
        a = np.array([1, 2, 3])
        col_vec = a[:,None]
        row_vec = a[None, :]
        return col_vec.shape, row_vec.shape
    _()
    return


@app.cell
def _(np):
    def _():
        A = np.array([[1, 2, 3],
                      [4, 5, 6]])
        b = np.array([10, 20])
        return A * b[:, None]
    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
