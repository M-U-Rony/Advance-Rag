/** Error types shared by the pipeline and the HTTP layer. */

/** The caller supplied unusable input (bad PDF, impossible chunking parameters...). Maps to HTTP 422. */
export class InvalidInputError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "InvalidInputError";
  }
}

/** An error that should be reported to the client as `{ "detail": message }` with `status`. */
export class HttpError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
  ) {
    super(detail);
    this.name = "HttpError";
  }
}
