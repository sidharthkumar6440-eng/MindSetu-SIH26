/**
 * MindSetu Unified API Service
 * Centralizes all backend network communication with timeout handling,
 * error normalization, and streaming NDJSON support.
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";
const DEFAULT_TIMEOUT_MS = 20000;
const SESSION_TIMEOUT_MS = 6000;

function createUUID() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/**
 * Deterministic client-side welfare triage calculation matching SIH26186 engine (AGENTS.md).
 * Used when backend triage endpoint is unreachable.
 */
export function calculateLocalTriage(answers, workload) {
  const safeAnswers = Array.isArray(answers) && answers.length === 6 ? answers : [1, 1, 1, 1, 1, 1];
  const ansSum = safeAnswers.reduce((acc, v) => acc + (typeof v === "number" ? Math.max(0, Math.min(3, v)) : 1), 0);
  const wellnessScore = Math.round(((ansSum / 18) * 100) * 100) / 100;

  const wl = workload || {};
  let wlScore = 0;
  const duty = Number(wl.duty_hours || 8);
  if (duty > 12) wlScore += 30;
  else if (duty > 10) wlScore += 20;
  else if (duty > 8) wlScore += 10;
  else wlScore += 5;

  const night = Number(wl.night_duties || 0);
  if (night > 4) wlScore += 20;
  else if (night > 2) wlScore += 15;
  else if (night > 0) wlScore += 10;

  const rest = Number(wl.rest_hours || 8);
  if (rest < 5) wlScore += 20;
  else if (rest < 6) wlScore += 15;
  else if (rest < 7) wlScore += 10;
  else wlScore += 5;

  const leaveGap = Number(wl.days_since_leave || 7);
  if (leaveGap > 45) wlScore += 15;
  else if (leaveGap > 30) wlScore += 10;
  else if (leaveGap > 14) wlScore += 5;

  const level = Math.max(1, Math.min(5, Number(wl.workload_level || 3)));
  wlScore += level * 2;

  if (Boolean(wl.high_pressure_assignment)) wlScore += 5;
  const changes = Number(wl.duty_change_frequency || 0);
  if (changes > 3) wlScore += 5;
  else if (changes > 0) wlScore += 3;

  const workloadScore = Math.min(100, Math.max(0, wlScore));
  const combinedScore = Math.round((0.55 * wellnessScore + 0.45 * workloadScore) * 100) / 100;

  let riskLevel = "low";
  if (combinedScore >= 70 || wellnessScore >= 80 || workloadScore >= 85) {
    riskLevel = "high";
  } else if (combinedScore >= 45 || wellnessScore >= 50 || workloadScore >= 60) {
    riskLevel = "moderate";
  }

  const recommendations = {
    high: "High cumulative stress and workload detected. Immediate rest, duty rotation review, and proactive check-in with a designated medical or welfare officer are strongly recommended.",
    moderate: "Moderate strain indicators observed. Prioritize scheduled recovery, sleep hygiene, and light physical de-escalation routines to prevent persistent exhaustion.",
    low: "Operational wellbeing indicators remain within stable baseline parameters. Continue scheduled rest breaks, peer engagement, and standard health routines.",
  };

  return {
    id: createUUID(),
    wellness_score: wellnessScore,
    workload_score: workloadScore,
    combined_score: combinedScore,
    risk_level: riskLevel,
    recommendation: recommendations[riskLevel],
    created_at: new Date().toISOString(),
    is_local_fallback: true,
  };
}

async function request(endpoint, options = {}, timeoutMs = DEFAULT_TIMEOUT_MS) {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
    });

    const text = await response.text();
    let data = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch {
      data = { detail: text };
    }

    if (!response.ok) {
      const errorMsg = data.detail || `Request failed with status ${response.status}`;
      throw new Error(errorMsg);
    }

    return data;
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error("Request timed out. Please check your connection and try again.", { cause: err });
    }
    throw err;
  } finally {
    window.clearTimeout(timeoutId);
  }
}

export const api = {
  // Session Management
  async createSession(consentGiven = true) {
    return request("/api/sessions", {
      method: "POST",
      body: JSON.stringify({ consent_given: consentGiven }),
    }, SESSION_TIMEOUT_MS);
  },

  generateFallbackSession(consentGiven = true) {
    return {
      session_id: createUUID(),
      anonymous: true,
      consent_given: consentGiven,
      is_fallback: true,
    };
  },

  // SIH26186 Welfare Workflow
  async submitWellness(sessionId, answers) {
    return request("/api/sih26186/wellness", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, answers }),
    });
  },

  async submitWorkload(sessionId, workloadData) {
    return request("/api/sih26186/workload", {
      method: "POST",
      body: JSON.stringify({
        session_id: sessionId,
        role: workloadData.role,
        unit: workloadData.unit || "",
        duty_hours: Number(workloadData.duty_hours),
        night_duties: Number(workloadData.night_duties),
        rest_hours: Number(workloadData.rest_hours),
        days_since_leave: Number(workloadData.days_since_leave),
        workload_level: Number(workloadData.workload_level),
        high_pressure_assignment: Boolean(workloadData.high_pressure_assignment),
        duty_change_frequency: Number(workloadData.duty_change_frequency || 0),
      }),
    });
  },

  async runAnalysis(sessionId) {
    return request(`/api/sih26186/analyze/${sessionId}`, {
      method: "POST",
    });
  },

  async getDashboard(sessionId) {
    return request(`/api/sih26186/dashboard/${sessionId}`);
  },

  async getAssessmentHistory(sessionId) {
    return request(`/api/sih26186/history/${sessionId}`);
  },


  // Explainable ML Research Assessment
  async runMLAnalyze(mlData) {
    return request("/api/sih26186/ml/analyze", {
      method: "POST",
      body: JSON.stringify({
        Q29_Total: Number(mlData.Q29_Total),
        Q12_weapon: Number(mlData.Q12_weapon),
        Q13_feltdie: Number(mlData.Q13_feltdie),
        Q23a_cutdowntime: Number(mlData.Q23a_cutdowntime),
        Q23b_Accomplished_less: Number(mlData.Q23b_Accomplished_less),
        Q23c_limited_work: Number(mlData.Q23c_limited_work),
        Q23d_difficulty_performing: Number(mlData.Q23d_difficulty_performing),
        generate_response: Boolean(mlData.generate_response ?? true),
      }),
    });
  },

  // Voice ML Analysis
  async analyzeVoice(sessionId, audioBase64) {
    return request("/api/sih26186/voice/analyze", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, audio_base64: audioBase64 }),
    });
  },

  async getVoiceDemoSample(scenario = "strained") {
    return request(`/api/sih26186/voice/demo-sample?scenario=${encodeURIComponent(scenario)}`, {
      method: "POST",
    });
  },

  // Doctor Connect & Appointment Scheduling
  async getDoctors({ specialization = "", availability = "" } = {}) {
    const params = new URLSearchParams();
    if (specialization) params.append("specialization", specialization);
    if (availability) params.append("availability", availability);
    const qs = params.toString() ? `?${params.toString()}` : "";
    return request(`/api/doctors${qs}`);
  },

  async getDoctor(doctorId) {
    return request(`/api/doctors/${doctorId}`);
  },

  async getDoctorAvailability(doctorId, dateStr) {
    return request(`/api/doctors/${doctorId}/availability?date=${encodeURIComponent(dateStr)}`);
  },

  async bookAppointment({ sessionId, doctorId, appointmentDate, appointmentTime, notes = "" }) {
    return request("/api/appointments", {
      method: "POST",
      body: JSON.stringify({
        session_id: sessionId,
        doctor_id: doctorId,
        appointment_date: appointmentDate,
        appointment_time: appointmentTime,
        notes: notes || null,
      }),
    });
  },

  async getAppointments(sessionId) {
    return request(`/api/appointments?session_id=${encodeURIComponent(sessionId)}`);
  },

  async getAppointment(appointmentId) {
    return request(`/api/appointments/${appointmentId}`);
  },

  async cancelAppointment(appointmentId) {
    return request(`/api/appointments/${appointmentId}/cancel`, {
      method: "POST",
    });
  },


  // Gemini AI Companion Streaming
  async streamChat({ sessionId, message, history, wellbeingContext, onToken, onComplete, onError }) {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 70000);

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          session_id: sessionId,
          message,
          history: (history || []).slice(-12).map((item) => ({ sender: item.sender, text: item.text })),
          wellbeing_context: wellbeingContext || {},
        }),
      });

      if (!response.ok) {
        const text = await response.text();
        let errMsg = `AI server error (${response.status})`;
        try {
          const parsed = JSON.parse(text);
          errMsg = parsed.detail || errMsg;
        } catch {
          // ignore
        }
        throw new Error(errMsg);
      }

      let fullContent = "";
      let isSafety = false;

      if (response.body && response.body.getReader) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;
            try {
              const chunk = JSON.parse(trimmed);
              if (chunk.type === "start" && chunk.risk_level === "safety_priority") {
                isSafety = true;
              }
              if (chunk.type === "token" && typeof chunk.content === "string") {
                fullContent += chunk.content;
                if (onToken) onToken(chunk.content);
              }
            } catch {
              // Ignore partial or non-JSON chunk lines
            }
          }
        }

        if (buffer.trim()) {
          try {
            const chunk = JSON.parse(buffer.trim());
            if (chunk.type === "token" && typeof chunk.content === "string") {
              fullContent += chunk.content;
              if (onToken) onToken(chunk.content);
            }
          } catch {
            // Ignore trailing partial buffer
          }
        }
      } else {
        const text = await response.text();
        const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
        for (const line of lines) {
          try {
            const chunk = JSON.parse(line);
            if (chunk.type === "start" && chunk.risk_level === "safety_priority") {
              isSafety = true;
            }
            if (chunk.type === "token" && typeof chunk.content === "string") {
              fullContent += chunk.content;
              if (onToken) onToken(chunk.content);
            }
          } catch {
            // Ignore non-NDJSON lines
          }
        }
      }

      if (!fullContent.trim()) {
        fullContent =
          "Thank you for sharing that. Take things one step at a time, and consider what practical recovery support would help most today. If your wellbeing feels difficult to manage, reaching out to someone you trust or a qualified professional can be a helpful next step.";
      }

      if (onComplete) onComplete({ text: fullContent.trim(), isSafety });
      return { text: fullContent.trim(), isSafety };
    } catch (err) {
      if (onError) onError(err);
      throw err;
    } finally {
      window.clearTimeout(timeoutId);
    }
  },

  // Mood Tracking
  async submitMood(sessionId, mood, note = null) {
    return request("/api/mood", {
      method: "POST",
      body: JSON.stringify({
        session_id: sessionId,
        mood: Number(mood),
        note: note ? note.trim() : null,
      }),
    });
  },

  async getMoodHistory(sessionId) {
    return request(`/api/mood/${sessionId}`);
  },

  async getMoodTrend() {
    return request("/api/dashboard/mood-trend");
  },
};
