/**
 * Embedder interface.
 *
 * An *embedder* maps text to a fixed-length vector so that semantically (or lexically)
 * similar texts end up close together. Two implementations ship with this project:
 * a from-scratch TF-IDF baseline and a real dense neural model.
 */
export interface Embedder {
  /** Short name shown in the UI and used as the vector cache directory. */
  readonly name: string;
  /** True if the embedder must see the whole corpus first (TF-IDF); false for pre-trained models. */
  readonly needsFit: boolean;
  /** Learn corpus statistics (only meaningful when `needsFit` is true). */
  fit(texts: string[]): void;
  /** Embed passages to one `Float32Array` row per text. */
  embedDocuments(texts: string[]): Promise<Float32Array[]>;
  /** Embed one query to a vector. */
  embedQuery(text: string): Promise<Float32Array>;
}
