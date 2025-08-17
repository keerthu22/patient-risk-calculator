import google.generativeai as genai

class PreventiveAdvisor:
    def __init__(self, api_key, model_option: str = "gemini-1.5-flash", temperature: float = 0.3):
        # Configure API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_option)
        self.temperature = temperature

    def generate_advice(self, patient_report: str) -> str:
        """
        Generate preventive care advice from a full patient risk report.
        Args:
            patient_report (str): The full formatted patient report as a string.
        Returns:
            str: Preventive care advice lines.
        """

        medical_prompt = f"""
        You are a medical AI assistant. Analyze the following patient risk report
        and generate concise, actionable **Preventive Care Advice**.

        The advice should be:
        - 3–6 short bullet points
        - Focused on prevention, lifestyle, and follow-up care
        - Avoid generic text, tailor advice to the diseases and risks in the report
        - Output only the advice, nothing else

        Patient Risk Report:
        {patient_report}
        """

        response = self.model.generate_content(
            [medical_prompt],
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=512,
            )
        )

        return response.text.strip()
