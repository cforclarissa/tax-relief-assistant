import streamlit as st

st.logo(
    image="images/reliefguide_logo.png",
    size="large"
)

st.title("About Us")

st.markdown("""
### Project
ReliefGuide: An AI-Powered Assistant for Accurate and Informed Tax Relief Claims

This project aims to support Singapore taxpayers in navigating and understanding the wide range of tax reliefs available to them. Today, taxpayers often struggle to locate accurate tax-relief information because it is distributed across multiple webpages and FAQ pages, which does not provide them an easy accessible way for them to obtain clear answers.
            
### Problem Statement

Taxpayers frequently experience uncertainty when determining if and how they can claim specific tax reliefs. They may not understand the conditions, required documents, or rules for reliefs such as Earned Income Relief, which is represented in the CSV.

Because information is scattered across multiple IRAS webpages, taxpayers spend unnecessary time searching for answers such as:
- What is this relief about?
- How much can I claim?
- What documents do I need?
- Is the relief automatically granted?

Without a unified system, taxpayers may miss out on reliefs or file incorrect claims, while call centre officers spend significant time handling repetitive FAQs.

How might we create a simple, unified, AI-powered assistant that helps taxpayers quickly retrieve accurate answers to common tax relief questions using structured FAQ data?

### Proposed Solution

The proposed solution is develop an AI-powered Tax Relief FAQ assistant built on the Retrieval-Augmented Generation (RAG) approach. 

1. Use the uploaded CSV as the Consolidated Dataset

The dataset contains:
- Relief categories
- Standardised questions
- Official answers

This structured format serves as the source material for retrieval.

2. Provide Intelligent, Conversational Q&A

The chatbot will allow taxpayers to ask questions such as:
- “What is Earned Income Relief?”
- “How do I qualify for this relief?”
- “What documents do I need?”

Using RAG, the system will retrieve the best-matching question/answer from the CSV and generate a conversational, easy-to-understand response.

3. Ensure Policy Accuracy and Consistency

Because responses are grounded in the CSV:
- The chatbot avoids hallucinations
- Answers remain aligned with approved content
- Updates can be easily made by modifying the CSV file

This makes the solution simple to maintain and scalable for future relief categories.
            
### Impact

With ReliefGuide, taxpayers can quickly access clear, concise answers to their tax relief questions without navigating multiple webpages.

Expected Benefits
- Time savings: Reduces search time from several minutes per question to seconds.
- Greater confidence: Users receive consistent, accurate, FAQ-based responses.
- Lower support load: Reduces repetitive enquiries handled by IRAS officers.
- Easy maintainability: New reliefs or revised answers can be added by updating the CSV file.

Because the system is grounded in structured, curated data, it supports high accuracy and reduces the risk of policy misinterpretation.
            
### Submission Tags

Capstone (Type 2)
         
### Data Classification and Sensitivity        
Data is classified to be Official Open / Non-Sensitive.       

### Sample Data Set             

The data collected are from the following publicly available links:
            
https://www.iras.gov.sg/taxes/individual-income-tax/basics-of-individual-income-tax/tax-reliefs-rebates-and-deductions/tax-reliefs/earned-income-relief

https://www.iras.gov.sg/taxes/individual-income-tax/basics-of-individual-income-tax/tax-reliefs-rebates-and-deductions/tax-reliefs/spouse-relief-spouse-relief-(disability)

https://www.iras.gov.sg/taxes/individual-income-tax/basics-of-individual-income-tax/tax-reliefs-rebates-and-deductions/tax-reliefs/foreign-domestic-worker-levy-(fdwl)-relief
            
https://www.iras.gov.sg/taxes/individual-income-tax/basics-of-individual-income-tax/tax-reliefs-rebates-and-deductions/tax-reliefs/central-provident-fund(cpf)-relief-for-employees

https://www.iras.gov.sg/taxes/individual-income-tax/basics-of-individual-income-tax/tax-reliefs-rebates-and-deductions/tax-reliefs/nsman-relief-(self-wife-and-parent)                          

https://www.iras.gov.sg/taxes/individual-income-tax/basics-of-individual-income-tax/tax-reliefs-rebates-and-deductions/tax-reliefs/parent-relief-parent-relief-(disability)

https://www.iras.gov.sg/taxes/individual-income-tax/basics-of-individual-income-tax/tax-reliefs-rebates-and-deductions/tax-reliefs/grandparent-caregiver-relief

https://www.iras.gov.sg/taxes/individual-income-tax/basics-of-individual-income-tax/tax-reliefs-rebates-and-deductions/tax-reliefs/sibling-relief-(disability)

https://www.iras.gov.sg/taxes/individual-income-tax/basics-of-individual-income-tax/tax-reliefs-rebates-and-deductions/tax-reliefs/working-mother's-child-relief-(wmcr)                        

https://www.iras.gov.sg/taxes/individual-income-tax/basics-of-individual-income-tax/tax-reliefs-rebates-and-deductions/tax-reliefs/qualifying-child-relief-(qcr)-child-relief-(disability)

https://www.iras.gov.sg/taxes/individual-income-tax/basics-of-individual-income-tax/tax-reliefs-rebates-and-deductions/tax-reliefs/life-insurance-relief            

https://www.iras.gov.sg/taxes/individual-income-tax/basics-of-individual-income-tax/tax-reliefs-rebates-and-deductions/tax-reliefs/course-fees-relief

https://www.iras.gov.sg/taxes/individual-income-tax/basics-of-individual-income-tax/special-tax-schemes/srs-contributions#title4

https://www.iras.gov.sg/taxes/individual-income-tax/basics-of-individual-income-tax/tax-reliefs-rebates-and-deductions/tax-reliefs/central-provident-fund-(cpf)-cash-top-up-relief

https://www.iras.gov.sg/taxes/individual-income-tax/basics-of-individual-income-tax/tax-reliefs-rebates-and-deductions/tax-reliefs/compulsory-and-voluntary-medisave-contributions           

### Team Member     
Clarissa Lo          


""")

with st.expander("Disclaimer"):
    
    # Add disclaimer with new lines
    disclaimer = """
    **IMPORTANT NOTICE**: This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.\n\nFurthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.\n\nAlways consult with qualified professionals for accurate and personalized advice. 
   
    """
    st.markdown(disclaimer)
