import os
import google.generativeai as genai
from typing import TypedDict, Annotated
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph


class VideoSummarizationState(TypedDict):
    ocr_text: str
    transcript_text: str
    ocr_summary: Annotated[str, "ocr_summary"]
    transcript_documentation: Annotated[str, "transcript_documentation"]
    final_output: Annotated[str, "final_output"]


class VideoSummarizer:
    def __init__(
        self,
        ocr_path: str,
        transcript_path: str,
        summary_out: str = "results/summary.md",
        full_doc_out: str = "results/full_doc.md",
    ):
        self.ocr_path = ocr_path
        self.transcript_path = transcript_path
        self.summary_out = summary_out
        self.full_doc_out = full_doc_out

        self.SYSTEM_CONTEXT = """
        You are an AI system tasked with processing and summarizing technical video content, using two data sources:
        - OCR text extracted from video frames, which may be incomplete, noisy, or unclear.
        - Audio transcription text extracted from the video's spoken content, which is assumed to be cleaner and more reliable.

        Your goals:
        1. For the OCR text: extract only the most useful information, clearly mention any uncertainties or partial information.
        2. For the transcription text: provide thorough, accurate, and clear full documentation.
        3. Combine both sources intelligently, using transcription as the primary basis and enhancing it with OCR insights where OCR data is reliable.
        4. Produce two output formats:
           - SUMMARY: A concise, clear summary of the combined information.
           - FULL DOCUMENTATION: A detailed document merging insights from both sources.
        """

        self.model = self._init_model()
        self.app = self._build_pipeline()

    def _init_model(self):
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=GOOGLE_API_KEY)
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            convert_system_message_to_human=True,
        )

    def _ocr_agent(self, data: VideoSummarizationState):
        messages = [
            SystemMessage(content=self.SYSTEM_CONTEXT),
            HumanMessage(content=f"""The following OCR text is extracted from video frames and may be noisy or incomplete:
\"\"\"{data.get("ocr_text", "")}\"\"\"
Extract the most useful information, clearly highlight any unclear or partial content.
Produce a brief summarized output indicating uncertainty or gaps if any.""")
        ]
        response = self.model.invoke(messages)
        return {"ocr_summary": response.content}

    def _transcription_agent(self, data: VideoSummarizationState):
        messages = [
            SystemMessage(content=self.SYSTEM_CONTEXT),
            HumanMessage(content=f"""Here is a transcript of the video's spoken content:
\"\"\"{data.get("transcript_text", "")}\"\"\"
Provide a thorough, accurate, and clear full documentation of this transcription.
Your output should be detailed, precise, and faithfully represent the original content,
suitable for use as a reference document.""")
        ]
        response = self.model.invoke(messages)
        return {"transcript_documentation": response.content}

    def _fusion_agent(self, data: VideoSummarizationState):
        ocr_summary = data.get("ocr_summary", "")
        transcript_doc = data.get("transcript_documentation", "")
        messages = [
            SystemMessage(content=self.SYSTEM_CONTEXT),
            HumanMessage(content=f"""You are given two sources of extracted video information:

    [OCR Summary]
    {ocr_summary}

    [Transcript Documentation]
    {transcript_doc}

    1. Create a concise SUMMARY of the video based on both sources.
    2. Then provide a comprehensive FULL DOCUMENTATION combining both.

    Respond in this format:

    ### SUMMARY
    <your summary>

    ### FULL DOCUMENTATION
    <your detailed documentation>
    """)
        ]
        response = self.model.invoke(messages)
        response_text = response.content

        try:
            _, summary_part = response_text.split("### SUMMARY", 1)
            summary_text, full_doc = summary_part.split("### FULL DOCUMENTATION", 1)
            summary_text = summary_text.strip()
            full_doc = full_doc.strip()
        except Exception:
            summary_text = "Error splitting summary."
            full_doc = response_text

        os.makedirs(os.path.dirname(self.summary_out), exist_ok=True)
        with open(self.summary_out, "w") as f:
            f.write(summary_text)
        with open(self.full_doc_out, "w") as f:
            f.write(full_doc)

        return {
            "summary": summary_text,
            "full_documentation": full_doc,
            "ocr_summary": ocr_summary,
            "transcript_doc": transcript_doc,
        }

    def _build_pipeline(self):
        builder = StateGraph(VideoSummarizationState)

        builder.add_node("ocr", self._ocr_agent)
        builder.add_node("transcription", self._transcription_agent)
        builder.add_node("fusion", self._fusion_agent)

        builder.set_entry_point("ocr")
        builder.add_edge("ocr", "transcription")
        builder.add_edge("transcription", "fusion")
        builder.set_finish_point("fusion")

        return builder.compile()

    def summarize(self):
        with open(self.ocr_path, "r") as f:
            ocr_text = f.read()
        with open(self.transcript_path, "r") as f:
            transcript_text = f.read()

        input_data = {
            "ocr_text": ocr_text,
            "transcript_text": transcript_text,
        }

        result = self.app.invoke(input_data)

        print("\n===== FINAL OUTPUT =====")
        print(f"\n> Summary saved to: {self.summary_out}")
        print(f"> Full documentation saved to: {self.full_doc_out}")

        return result
