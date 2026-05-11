import contextvars

rephrased_query_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "rephrased_query", default=None
)
