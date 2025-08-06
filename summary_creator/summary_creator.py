from langchain_core.messages import SystemMessage, HumanMessage
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated

def summarize_video(
    ocr_path="/home/salma/Projects/videoSummarizerAI/video_texts/AI_texts.txt",
    transcript_path="/home/salma/Projects/videoSummarizerAI/transcriptions/AI_transcription.txt",
    summary_out="results/summary.md",
    full_doc_out="results/full_doc.md"
):
    # --- State Definition ---
    class VideoSummarizationState(TypedDict):
        ocr_text: str
        transcript_text: str
        ocr_summary: Annotated[str, "ocr_summary"]
        transcript_documentation: Annotated[str, "transcript_documentation"]
        final_output: Annotated[str, "final_output"]




    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0,  convert_system_message_to_human=True)

    SYSTEM_CONTEXT = """
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

    # === OCR agent node ===
    def ocr_agent(data: VideoSummarizationState):
        ocr_text = data.get("ocr_text", "")
        messages = [
            SystemMessage(content=SYSTEM_CONTEXT),
            HumanMessage(content=f"""The following OCR text is extracted from video frames and may be noisy or incomplete:
\"\"\"{ocr_text}\"\"\"
Extract the most useful information, clearly highlight any unclear or partial content.
Produce a brief summarized output indicating uncertainty or gaps if any.""")
        ]
        response = model.invoke(messages)
        return {"ocr_summary": response.content}

    # === Transcription agent node ===
    def transcription_agent(data: VideoSummarizationState):
        transcript_text = data.get("transcript_text", "")
        messages = [
            SystemMessage(content=SYSTEM_CONTEXT),
            HumanMessage(content=f"""Here is a transcript of the video's spoken content:
\"\"\"{transcript_text}\"\"\"
Provide a thorough, accurate, and clear full documentation of this transcription.
Your output should be detailed, precise, and faithfully represent the original content,
suitable for use as a reference document.""")
        ]
        response = model.invoke(messages)
        return {"transcript_documentation": response.content}

    # === Fusion node ===
    def fusion_agent(data: VideoSummarizationState):
        ocr_summary = data.get("ocr_summary", "")
        transcript_doc = data.get("transcript_documentation", "")
        messages = [
            SystemMessage(content=SYSTEM_CONTEXT),
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
        response = model.invoke(messages)
        response_text = response.content

        # Split the response into summary and full doc
        try:
            _, summary_part = response_text.split("### SUMMARY", 1)
            summary_text, full_doc = summary_part.split("### FULL DOCUMENTATION", 1)
            summary_text = summary_text.strip()
            full_doc = full_doc.strip()
        except Exception as e:
            summary_text = "Error splitting summary."
            full_doc = response_text

        # Write to files
        os.makedirs(os.path.dirname(summary_out), exist_ok=True)
        with open(summary_out, "w") as f:
            f.write(summary_text)
        with open(full_doc_out, "w") as f:
            f.write(full_doc)


        final_output = {
            "summary": summary_text,
            "full_documentation": full_doc,
            "ocr_summary": ocr_summary,
            "transcript_doc": transcript_doc,
        }

        
        return final_output

    # === LangGraph setup ===
    builder = StateGraph(VideoSummarizationState)

    builder.add_node("ocr", ocr_agent)
    builder.add_node("transcription", transcription_agent)
    builder.add_node("gemini", fusion_agent)

    builder.set_entry_point("ocr")
    builder.add_edge("ocr", "transcription")
    builder.add_edge("transcription", "gemini")
    builder.set_finish_point("gemini")

    app = builder.compile()

    # === Load input from text files ===
    with open(ocr_path) as f:
        ocr_text = f.read()

    with open(transcript_path) as f:
        transcript_text = f.read()

    input_data = {
        "ocr_text": ocr_text,
        "transcript_text": transcript_text,
    }

    # === Run the graph ===
    final_output = app.invoke(input_data)

    # === Print output locations ===
    print("\n===== FINAL OUTPUT =====")
    print(f"\n> Summary saved to: {summary_out}")
    print(f"> Full documentation saved to: {full_doc_out}")

    return final_output