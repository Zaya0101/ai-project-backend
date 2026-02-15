import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import OpenAI from "openai";
import { InferenceClient } from "@huggingface/inference";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 999;

app.use(cors());
app.use(express.json({ limit: "50mb" })); // base64 зураг том байж болно

// Groq Setup (text LLM)
const groq = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: "https://api.groq.com/openai/v1",
});

// Hugging Face Router Setup (vision chat)
const hfRouter = new OpenAI({
  apiKey: process.env.HF_TOKEN,
  baseURL: "https://router.huggingface.co/v1",
});

// Hugging Face InferenceClient (text-to-image гэх мэт)
const hf = new InferenceClient(process.env.HF_TOKEN);

/* 1) IMAGE CAPTION (Image -> short text) */
app.post("/caption", async (req, res) => {
  const { image } = req.body; // data:image/...;base64,...

  try {
    if (!image || typeof image !== "string") {
      return res.status(400).json({ error: "Missing image" });
    }

    const completion = await hfRouter.chat.completions.create({
      model: "CohereLabs/aya-vision-32b:cohere",
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "Describe this food photo in ONE short sentence." },
            { type: "image_url", image_url: { url: image } },
          ],
        },
      ],
      max_tokens: 120,
    });

    const caption = completion.choices?.[0]?.message?.content?.trim() || "No result";
    return res.json({ caption });
  } catch (err) {
    console.error("❌ CAPTION ERROR:", err?.response?.data || err?.message || err);
    return res.status(500).json({ error: "Caption failed" });
  }
});

/* 2) INGREDIENTS FROM IMAGE (Image -> ingredients bullets) */
app.post("/ingredients-from-image", async (req, res) => {
  const { image } = req.body; // data:image/...;base64,...

  try {
    if (!image || typeof image !== "string") {
      return res.status(400).json({ error: "Missing image" });
    }

    const completion = await hfRouter.chat.completions.create({
      model: "CohereLabs/aya-vision-32b:cohere",
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text:
                "You are a food expert.\n" +
                "From this food photo, list the likely ingredients.\n" +
                "Return ONLY bullet points. No extra text. If unsure, include likely ingredients.",
            },
            { type: "image_url", image_url: { url: image } },
          ],
        },
      ],
      max_tokens: 220,
    });

    const ingredients =
      completion.choices?.[0]?.message?.content?.trim() || "- No result";

    return res.json({ ingredients });
  } catch (err) {
    console.error("❌ ING_FROM_IMAGE ERROR:", err?.response?.data || err?.message || err);
    return res.status(500).json({ error: "Failed to analyze image" });
  }
});

/* 3) INGREDIENTS (Text -> ingredients bullets) */
app.post("/ingredients", async (req, res) => {
  const { text } = req.body;

  try {
    const response = await groq.chat.completions.create({
      model: "llama-3.3-70b-versatile",
      messages: [
        { role: "system", content: "You are a food expert. List common ingredients as bullet points only." },
        { role: "user", content: text || "" },
      ],
    });

    return res.json({ ingredients: response.choices?.[0]?.message?.content || "- No result" });
  } catch (err) {
    console.error("INGREDIENT ERROR 👉", err?.response?.data || err?.message || err);
    return res.status(500).json({ error: "Failed to recognize ingredients" });
  }
});

/* 4) CHAT (AI-тай харилцах) */
app.post("/chat", async (req, res) => {
  const { messages } = req.body;

  try {
    const normalized = (messages || []).map((m) => ({
      role:
        m.role === "assistant" || m.role === "ai" ? "assistant" :
        m.role === "system" ? "system" :
        "user",
      content: m.content ?? m.text ?? "",
    }));

    const response = await groq.chat.completions.create({
      model: "llama-3.3-70b-versatile",
      messages: normalized,
    });

    return res.json({ reply: response.choices?.[0]?.message?.content || "No result" });
  } catch (err) {
    console.error("CHAT ERROR 👉", err?.response?.data || err?.message || err);
    return res.status(500).json({ error: "Chat failed" });
  }
});

/* 5) IMAGE CREATE (Зураг үүсгэх) */
app.post("/image-create", async (req, res) => {
  const { prompt } = req.body;

  try {
    const imageBlob = await hf.textToImage({
      model: "black-forest-labs/FLUX.1-schnell",
      inputs: prompt || "",
    });

    const buffer = Buffer.from(await imageBlob.arrayBuffer());
    const base64 = buffer.toString("base64");
    return res.json({ image: `data:image/png;base64,${base64}` });
  } catch (err) {
    console.error("HF ERROR 👉", err?.message || err);
    return res.status(500).json({ message: "Image generation failed" });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
});
