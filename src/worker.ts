interface MicroexpressionAnalysis {
  timestamp: number;
  agentId: string;
  confidence: number;
  sentiment: number;
  styleMetrics: {
    verbosity: number;
    formality: number;
    emotionality: number;
    certainty: number;
  };
  anomalies: string[];
  fingerprintHash: string;
}

interface AgentFingerprint {
  agentId: string;
  baseline: {
    avgSentiment: number;
    avgConfidence: number;
    styleProfile: number[];
  };
  driftScore: number;
  lastUpdated: number;
}

class BehavioralAnalyzer {
  private fingerprints: Map<string, AgentFingerprint> = new Map();
  private readonly DRIFT_THRESHOLD = 0.15;
  private readonly ANOMALY_CONFIDENCE_THRESHOLD = 0.3;

  analyzeOutput(agentId: string, output: string): MicroexpressionAnalysis {
    const timestamp = Date.now();
    const confidence = this.calculateConfidence(output);
    const sentiment = this.analyzeSentiment(output);
    const styleMetrics = this.extractStyleMetrics(output);
    const anomalies = this.detectAnomalies(output, confidence, sentiment);
    const fingerprintHash = this.generateFingerprintHash(agentId, styleMetrics);

    this.updateFingerprint(agentId, {
      sentiment,
      confidence,
      styleMetrics
    });

    return {
      timestamp,
      agentId,
      confidence,
      sentiment,
      styleMetrics,
      anomalies,
      fingerprintHash
    };
  }

  private calculateConfidence(output: string): number {
    const indicators = {
      hedging: (output.match(/\b(maybe|perhaps|possibly|might)\b/gi) || []).length,
      certainty: (output.match(/\b(certainly|definitely|absolutely|clearly)\b/gi) || []).length,
      qualifiers: (output.match(/\b(somewhat|quite|rather|fairly)\b/gi) || []).length
    };

    const totalWords = output.split(/\s+/).length || 1;
    const hedgeScore = indicators.hedging / totalWords;
    const certaintyScore = indicators.certainty / totalWords;
    
    return Math.max(0, Math.min(1, 0.5 + (certaintyScore * 2) - (hedgeScore * 3)));
  }

  private analyzeSentiment(output: string): number {
    const positive = (output.match(/\b(good|great|excellent|positive|happy|success)\b/gi) || []).length;
    const negative = (output.match(/\b(bad|poor|negative|unhappy|fail|error)\b/gi) || []).length;
    const total = positive + negative || 1;
    
    return (positive - negative) / total;
  }

  private extractStyleMetrics(output: string) {
    const words = output.split(/\s+/);
    const sentences = output.split(/[.!?]+/).filter(s => s.trim().length > 0);
    
    return {
      verbosity: words.length / Math.max(sentences.length, 1),
      formality: this.calculateFormality(output),
      emotionality: this.calculateEmotionality(output),
      certainty: this.calculateCertainty(output)
    };
  }

  private calculateFormality(text: string): number {
    const formal = (text.match(/\b(therefore|however|moreover|furthermore|consequently)\b/gi) || []).length;
    const informal = (text.match(/\b(hey|okay|cool|awesome|stuff|thing)\b/gi) || []).length;
    const total = formal + informal || 1;
    
    return formal / total;
  }

  private calculateEmotionality(text: string): number {
    const emotional = (text.match(/!\s*$|(\b(amazing|terrible|wow|ugh|alas)\b)/gi) || []).length;
    const totalSentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0).length || 1;
    
    return emotional / totalSentences;
  }

  private calculateCertainty(text: string): number {
    const strong = (text.match(/\b(must|will|shall|certainly|definitely)\b/gi) || []).length;
    const weak = (text.match(/\b(might|could|perhaps|maybe|possibly)\b/gi) || []).length;
    const total = strong + weak || 1;
    
    return strong / total;
  }

  private detectAnomalies(output: string, confidence: number, sentiment: number): string[] {
    const anomalies: string[] = [];

    if (confidence < this.ANOMALY_CONFIDENCE_THRESHOLD) {
      anomalies.push("low_confidence");
    }

    if (Math.abs(sentiment) > 0.8) {
      anomalies.push("extreme_sentiment");
    }

    const questionCount = (output.match(/\?/g) || []).length;
    if (questionCount > 3) {
      anomalies.push("excessive_questioning");
    }

    const repetition = this.checkRepetition(output);
    if (repetition) {
      anomalies.push("repetitive_patterns");
    }

    return anomalies;
  }

  private checkRepetition(text: string): boolean {
    const words = text.toLowerCase().split(/\s+/);
    const wordCounts: Map<string, number> = new Map();
    
    words.forEach(word => {
      wordCounts.set(word, (wordCounts.get(word) || 0) + 1);
    });

    const totalWords = words.length;
    for (const count of wordCounts.values()) {
      if (count / totalWords > 0.1) {
        return true;
      }
    }
    
    return false;
  }

  private generateFingerprintHash(agentId: string, styleMetrics: any): string {
    const data = `${agentId}:${JSON.stringify(styleMetrics)}:${Date.now()}`;
    let hash = 0;
    
    for (let i = 0; i < data.length; i++) {
      hash = ((hash << 5) - hash) + data.charCodeAt(i);
      hash |= 0;
    }
    
    return Math.abs(hash).toString(16).substring(0, 8);
  }

  private updateFingerprint(agentId: string, metrics: any) {
    const existing = this.fingerprints.get(agentId);
    const now = Date.now();
    
    if (!existing) {
      this.fingerprints.set(agentId, {
        agentId,
        baseline: {
          avgSentiment: metrics.sentiment,
          avgConfidence: metrics.confidence,
          styleProfile: [
            metrics.styleMetrics.verbosity,
            metrics.styleMetrics.formality,
            metrics.styleMetrics.emotionality,
            metrics.styleMetrics.certainty
          ]
        },
        driftScore: 0,
        lastUpdated: now
      });
      return;
    }

    const newProfile = [
      metrics.styleMetrics.verbosity,
      metrics.styleMetrics.formality,
      metrics.styleMetrics.emotionality,
      metrics.styleMetrics.certainty
    ];

    const drift = this.calculateDrift(existing.baseline.styleProfile, newProfile);
    
    existing.baseline.styleProfile = existing.baseline.styleProfile.map((val, idx) =>
      (val * 0.9) + (newProfile[idx] * 0.1)
    );
    
    existing.baseline.avgSentiment = (existing.baseline.avgSentiment * 0.9) + (metrics.sentiment * 0.1);
    existing.baseline.avgConfidence = (existing.baseline.avgConfidence * 0.9) + (metrics.confidence * 0.1);
    existing.driftScore = drift;
    existing.lastUpdated = now;
  }

  private calculateDrift(baseline: number[], current: number[]): number {
    let sum = 0;
    for (let i = 0; i < baseline.length; i++) {
      sum += Math.abs(baseline[i] - current[i]);
    }
    return sum / baseline.length;
  }

  getFingerprint(agentId: string): AgentFingerprint | null {
    return this.fingerprints.get(agentId) || null;
  }

  getDriftAlerts(): Array<{agentId: string, driftScore: number}> {
    const alerts: Array<{agentId: string, driftScore: number}> = [];
    
    this.fingerprints.forEach((fingerprint, agentId) => {
      if (fingerprint.driftScore > this.DRIFT_THRESHOLD) {
        alerts.push({
          agentId,
          driftScore: fingerprint.driftScore
        });
      }
    });
    
    return alerts;
  }
}

const analyzer = new BehavioralAnalyzer();

async function handleRequest(request: Request): Promise<Response> {
  const url = new URL(request.url);
  const corsHeaders = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type"
  };

  if (request.method === "OPTIONS") {
    return new Response(null, {
      headers: corsHeaders
    });
  }

  if (url.pathname === "/api/analyze" && request.method === "POST") {
    try {
      const { agentId, output } = await request.json();
      
      if (!agentId || !output) {
        return new Response(JSON.stringify({ error: "Missing agentId or output" }), {
          status: 400,
          headers: {
            "Content-Type": "application/json",
            ...corsHeaders
          }
        });
      }

      const analysis = analyzer.analyzeOutput(agentId, output);
      
      return new Response(JSON.stringify(analysis), {
        headers: {
          "Content-Type": "application/json",
          "X-Content-Type-Options": "nosniff",
          "X-Frame-Options": "DENY",
          ...corsHeaders
        }
      });
    } catch (error) {
      return new Response(JSON.stringify({ error: "Invalid request" }), {
        status: 400,
        headers: {
          "Content-Type": "application/json",
          ...corsHeaders
        }
      });
    }
  }

  if (url.pathname.startsWith("/api/fingerprint/") && request.method === "GET") {
    const agentId = url.pathname.split("/").pop();
    
    if (!agentId) {
      return new Response(JSON.stringify({ error: "Missing agentId" }), {
        status: 400,
        headers: {
          "Content-Type": "application/json",
          ...corsHeaders
        }
      });
    }

    const fingerprint = analyzer.getFingerprint(agentId);
    
    if (!fingerprint) {
      return new Response(JSON.stringify({ error: "Fingerprint not found" }), {
        status: 404,
        headers: {
          "Content-Type": "application/json",
          ...corsHeaders
        }
      });
    }

    return new Response(JSON.stringify(fingerprint), {
      headers: {
        "Content-Type": "application/json",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        ...corsHeaders
      }
    });
  }

  if (url.pathname === "/api/drift" && request.method === "GET") {
    const alerts = analyzer.getDriftAlerts();
    
    return new Response(JSON.stringify({ alerts }), {
      headers: {
        "Content-Type": "application/json",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        ...corsHeaders
      }
    });
  }

  if (url.pathname === "/health") {
    return new Response(JSON.stringify({ status: "healthy", service: "agent-microexpressions" }), {
      headers: {
        "Content-Type": "application/json",
        ...corsHeaders
      }
    });
  }

  return new Response(JSON.stringify({ error: "Not found" }), {
    status: 404,
    headers: {
      "Content-Type": "application/json",
      ...corsHeaders
    }
  });
}

export default {
  async fetch(request: Request): Promise<Response> {
    return handleRequest(request);
  }
};
