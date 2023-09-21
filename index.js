import { LlamaCpp } from "langchain/llms/llama_cpp";
import promptSync from "prompt-sync";

const llamaPath = "./7b-chat/ggml-model-q4_0.gguf";
const prompt = promptSync({
  sigint: true,
});

const model = new LlamaCpp({
  modelPath: llamaPath,
  contextSize: 2048,
  batchSize: 256,
  topK: 10000,
  temperature: 0.2,
  maxConcurrency: 8,
});

while (true) {
  const question = prompt("You: ");
  if (question === "exit") {
    process.exit();
  }
  const response = await model.call(question);
  console.log(`AI: ${response}`);
}
