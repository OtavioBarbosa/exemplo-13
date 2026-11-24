import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { type TextSplitterConfig } from "./config.ts";

export class DocumentProcessor {
    private pdf_path: string;
    private text_splitter_config: TextSplitterConfig;

    constructor(pdf_path: string, text_splitter_config: TextSplitterConfig) {
        this.pdf_path = pdf_path;
        this.text_splitter_config = text_splitter_config;
    }

    async loadAndSplit(): Promise<{ pageContent: string; metadata: { source: string } }[]> {
        const loader = new PDFLoader(this.pdf_path);
        const raw_documents = await loader.load();
        const splitter = new RecursiveCharacterTextSplitter(
            this.text_splitter_config
        );
        const documents = await splitter.splitDocuments(raw_documents);
        return documents.map(doc => ({
            ...doc,
            metadata: {
                source: doc.metadata.source,
            },
        }));
    }
}