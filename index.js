import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import OpenAI from "openai";
import { InferenceClient } from "@huggingface/inference";

dotenv.config();

const app = express();
// Портыг хатуу 999 дээр эсвэл env-ээс авна
const PORT = process.env.PORT || 999;

app.use(cors());
app.use(express.json({ limit: "50mb" }));

// 🛡️ API Key шалгах
if (!process.env.OPENAI_API_KEY || !process.env.HF_TOKEN) {
  console.error("❌ АЛДАА: .env файл дотор API Key-үүд дутуу байна!");
}

const groq = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: "https://api.groq.com/openai/v1",
});

const hfRouter = new OpenAI({
  apiKey: process.env.HF_TOKEN,
  baseURL: "https://router.huggingface.co/v1",
});

const hf = new InferenceClient(process.env.HF_TOKEN);

// ---------------------------------------------------------
// 1) TEXT INGREDIENTS (Frontend-ийн IngredientRecognitionTab-д зориулав)
// ---------------------------------------------------------
app.post("/ingredients", async (req, res) => {
  const { text } = req.body;
  if (!text) return res.status(400).json({ error: "Text is required" });

  try {
    const response = await groq.chat.completions.create({
      model: "llama-3.3-70b-versatile",
      messages: [
        {
          role: "system",
          content:
            "You are a food expert. List common ingredients as bullet points only.",
        },
        { role: "user", content: text },
      ],
    });
    return res.json({
      ingredients: response.choices?.[0]?.message?.content || "No result",
    });
  } catch (err) {
    console.error("INGREDIENT ERROR 👉", err.message);
    return res.status(500).json({ error: "Failed to recognize ingredients" });
  }
});

// ---------------------------------------------------------
// 2) IMAGE INGREDIENTS (Frontend-ийн ImageUpload/inputFile-д зориулав)
// ---------------------------------------------------------
app.post("/ingredients-from-image", async (req, res) => {
  const { image } = req.body;
  if (!image) return res.status(400).json({ error: "Missing image" });

  try {
    const completion = await hfRouter.chat.completions.create({
      model: "CohereLabs/aya-vision-32b:cohere",
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text: "You are a food expert. From this food photo, list the likely ingredients. Return ONLY bullet points.",
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
    console.error("❌ IMAGE_ING ERROR:", err.message);
    return res.status(500).json({ error: "Failed to analyze image" });
  }
});

// ---------------------------------------------------------
// 3) CHAT & 4) IMAGE CREATE (Бусад замууд хэвээрээ)
// ---------------------------------------------------------
app.post("/chat", async (req, res) => {
  const { messages } = req.body;
  try {
    const normalized = (messages || []).map((m) => ({
      role: m.role === "ai" ? "assistant" : m.role,
      content: m.text || m.content || "",
    }));
    const response = await groq.chat.completions.create({
      model: "llama-3.3-70b-versatile",
      messages: normalized,
    });
    return res.json({
      reply: response.choices?.[0]?.message?.content || "No result",
    });
  } catch (err) {
    return res.status(500).json({ error: "Chat failed" });
  }
});

app.post("/image-create", async (req, res) => {
  const { prompt } = req.body;
  try {
    const imageBlob = await hf.textToImage({
      model: "black-forest-labs/FLUX.1-schnell",
      inputs: prompt || "",
    });
    const buffer = Buffer.from(await imageBlob.arrayBuffer());
    return res.json({
      image: `data:image/png;base64,${buffer.toString("base64")}`,
    });
  } catch (err) {
    return res.status(500).json({ message: "Image generation failed" });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
});
