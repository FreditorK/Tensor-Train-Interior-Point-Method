import jax


def partial_D(f, idx):
    return jax.jit(jax.vmap(jax.grad(f, argnums=idx), -1))


def D_func(f):
    return jax.jit(jax.vmap(jax.grad(f), -1))


def J_func(f, argnums=0):
    return jax.jit(jax.jacfwd(f, argnums=argnums))
