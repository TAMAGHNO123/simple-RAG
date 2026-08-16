export interface AskRequest {
  question: string;
  top_k?: number;
}

export interface AskResponse {
  answer: string;
  sources: string[];
}

export type TurnStatus = "pending" | "done" | "error";

export interface Turn {
  question: string;
  answer: string;
  sources: string[];
  status: TurnStatus;
}

export interface HealthResponse {
  status: string;
  vectors: number;
  model: string;
}