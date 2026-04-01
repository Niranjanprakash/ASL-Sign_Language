/**
 * api.js - Backend communication layer.
 * Sends 63 landmark floats to Flask /predict endpoint.
 */

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:5000";

/**
 * Send landmarks to backend and receive prediction.
 * @param {number[]} landmarks - Array of 63 floats (21 landmarks × [x,y,z])
 * @returns {Promise<{prediction, confidence, status, confusion, top2}>}
 */
export async function predictASL(landmarks) {
  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ landmarks }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${response.status}`);
  }
  return response.json();
}

/**
 * Health-check the backend.
 */
export async function healthCheck() {
  const response = await fetch(`${API_BASE}/health`);
  return response.json();
}
