import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { CONFIG } from "./config.ts";
import { DocumentProcessor } from "./document_processor.ts";
import { type PretrainedOptions } from "@huggingface/transformers";
import { Neo4jVectorStore } from "@langchain/community/vectorstores/neo4j_vector";
import { displayResults } from "./util.ts";
import { ChatOpenAI } from "@langchain/openai";
import { AI } from "./ai.ts";
import { writeFile, mkdir } from 'node:fs/promises'

let _ne04j_vector_store: any = null;

async function clearAll(vector_store: Neo4jVectorStore, node_label: string) {
    await vector_store.query(`MATCH (n:${node_label}) DETACH DELETE n`);
}

try { 
    const documentProcessor = new DocumentProcessor(
        CONFIG.pdf.path,
        CONFIG.textSplitter
    );

    const documents = await documentProcessor.loadAndSplit();
    const embeddings = new HuggingFaceTransformersEmbeddings({
        model: CONFIG.embedding.modelName,
        pretrainedOptions: CONFIG.embedding.pretrainedOptions as PretrainedOptions,
    });

    const nlp_model = new ChatOpenAI({
        openAIApiKey: CONFIG.openRouter.apiKey,
        modelName: CONFIG.openRouter.nlpModel,
        temperature: CONFIG.openRouter.temperature,
        maxRetries: CONFIG.openRouter.maxRetries,
        configuration: {
            baseURL: CONFIG.openRouter.url,
            defaultHeaders: CONFIG.openRouter.defaultHeaders,
        }
    });

    _ne04j_vector_store = await Neo4jVectorStore.fromExistingGraph(
        embeddings,
        CONFIG.neo4j,
    );

    clearAll(_ne04j_vector_store, CONFIG.neo4j.nodeLabel);

    for (const document of documents) {
        await _ne04j_vector_store.addDocuments([document])
    }

    const questions = [
        "O que significa treinar uma rede neural?",
    ];

    const ai = new AI({    
        nlpModel: nlp_model,
        debugLog: console.log,
        vectorStore: _ne04j_vector_store,
        promptConfig: CONFIG.promptConfig,
        templateText: CONFIG.templateText,
        topK: CONFIG.similarity.topK,
    });

    for (const index in questions) {
        const question = questions[index]
        console.log(`\n${'='.repeat(80)}`);
        console.log(`📌 PERGUNTA: ${question}`);
        console.log('='.repeat(80));
        const result = await ai.answerQuestion(question!)
        if(result.error) {
            console.log(`\n❌ Erro: ${result.error}\n`);
            continue
        }

        console.log(`\n${result.answer}\n`);
        await mkdir(CONFIG.output.answersFolder, { recursive: true })

        const fileName = `${CONFIG.output.answersFolder}/${CONFIG.output.fileName}-${index}-${Date.now()}.md`

        await writeFile(fileName, result.answer!)
    }


    // Cleanup
    console.log(`\n${'='.repeat(80)}`);
    console.log("✅ Processamento concluído com sucesso!\n");
}
catch (error) {
    console.error("An error occurred:", error);
}
finally {
    await _ne04j_vector_store?.close();
}