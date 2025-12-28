import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import backend

#setting page configuration
st.set_page_config(page_title="Resume Analyser",page_icon="📄",layout="wide",
                   initial_sidebar_state="expanded")

#custom css
st.markdown("""
    <style>
        .metric-card{
            background-color: #f9f9f9;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
        }
        .stProgress > div> div > div > div{
            background-image: linear-gradient(to right,#ff4b4b,#fca311,#4caf50);    
        }   
    </style>
""",unsafe_allow_html=True)

#Main Page
st.title(" Smart Resume Analyser")
st.markdown("### Optimize your Resume for ATS with AI-Powered Insights")

#Sidebar Designing
with st.sidebar:
    st.header(" How it Works ")
    st.markdown("""
    1. **Context Match:** Uses AI to understand the *meaning* of your Experience.
    2. **Skill Match:** Checks for specific technical Skills required.
    3. **Vocabulary:** Checks if you use the right industry jargons.
    """)
    st.markdown("---")
    st.caption("Build using SBERT, TF-IDF & Keyword Matching")
#Taking input
col1,col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type="pdf")
with col2:
    job_description = st.text_area("Paste the Job description",height = 200,
                                    placeholder="Paste the Whole Job Description Here...")
#Execution
if st.button("Analyse the Resume",type="primary"):
    if uploaded_file and job_description:
        with st.spinner("Reading your Resume... Extracting Skills... Calculating Scores... "):
            
            #Preprocessing
            resume_text = backend.extract_text_from_pdf(uploaded_file)
            contact = backend.contact_info(resume_text)
            sbert = backend.semantic_matching_Score(resume_text,job_description)
            tfidf = backend.match_score(resume_text,job_description)
            keyword_score,missing,matched = backend.keyword_matched(resume_text,job_description)
            final_score,feedback = backend.generate_analysis_report(
                    sbert,
                    tfidf,
                    keyword_score,
                    matched,
                    missing
                )
            #Contact details
            st.subheader("Candidate Profile")
            with st.container(border=True):
                col1, col2, col3 = st.columns(3)
                col1.markdown(f"**Email ID**\n\n{contact.get('email_id', 'N/A')}")
                col2.markdown(f"**Mobile No**\n\n{contact.get('Phone_no', 'N/A')}")
    
                with col3:
                    st.markdown("**Portfolio / Links**")
                    links = contact.get("Links", [])
                    if links:
                        with st.expander(f"{len(links)} Link(s) Found"):
                            for i, link in enumerate(links):
                                display_text = link[:40] + "..." if len(link) > 40 else link
                                st.markdown(f"**Link {i+1}:** [{display_text}]({link})")
                    else:
                        st.caption("No links detected")
            st.divider()
            #Part 2
            col_left,col_right = st.columns([1,2])

            with col_left:
                st.markdown("#### Overall Match Distribution!")
                if final_score >= 60:
                    chart_color = "#4CAF50"
                elif final_score >= 40:
                    chart_color = "#ffa726"
                else:
                    chart_color = "#ef5350"
                fig = go.Figure(data=[go.Pie(
                    labels = ['Match Score','Gap'],
                    values = [final_score,100-final_score],
                    hole = .7,
                    marker=dict(colors = [chart_color,'#e0e0e0']),
                    textinfo='none',
                    hoverinfo='label+percent',
                    sort=False
                )])
                fig.update_layout(
                    showlegend = False,
                    margin = dict(t=0,b=0,l=0,r=0),
                    height = 250,
                    annotations = [dict(text=f"{final_score}%",x=0.5,y=0.5,
                                        font_size=45,font_weight= "bold",showarrow = False)]
                )
                st.plotly_chart(fig,use_container_width=True)
                   
                #Sumarrised verdict
                if final_score>60:
                    st.success("Verdict : **STRONG MATCH**")
                elif final_score>=50:
                    st.warning("Verdict: **POTENTIAL MATCH**")
                else:
                    st.error("Verdict: **LOW MATCH**")
                    
            with col_right:
                st.markdown("##### Detailed Metrics Breakdown")
                m1,m2,m3 = st.columns(3)
                m1.metric("Contextual Match(AI)",f"{sbert}%","60% weight")
                m2.metric("Hard Skills",f"{keyword_score}%","30% weight")
                m3.metric("Vocabulary",f"{tfidf}%","10% weight")

                contribution_data = {
                'Semantic Score': sbert * 0.6,
                'Keywords Matched': keyword_score * 0.3,
                'Vocabulary': tfidf * 0.1
                }
                pie_chart = go.Figure(data=[go.Pie(
                    labels=list(contribution_data.keys()),
                    values=list(contribution_data.values()),
                    hole=.4,
                    marker=dict(colors=['#5c6bc0', '#26a69a', '#ffa726']),
                    textinfo='label+percent',
                    showlegend=False
                 )])
    
                pie_chart.update_layout(
                margin=dict(t=45, b=0, l=0, r=0),
                height=330,
                title=dict(text="Score Contribution", font=dict(size=32), x=0.5,y=0.95,
                           xanchor = "center",yanchor = "top")
                )
                st.plotly_chart(pie_chart, use_container_width=True)

                st.markdown("#### AI Feedback Analysis")
                for item in feedback:
                    st.markdown(f"- {item}")
                
            #Part 3
            st.divider()
            st.subheader("SKill Gap Analysis")
            sg1,sg2 = st.columns(2)
            with sg1:
                st.markdown(" #### Matched Skills !!")
                if matched:
                    st.write(", ".join([f"`{s}`"for s in matched]))
                else:
                    st.caption("No specific hard skill matched from the JD.")
            with sg2:
                st.markdown(" #### Missing Skills (Critical)")
                if missing:
                    for s in missing:
                        st.error(f"Missing: **{s.replace('-',' ').title()}**")
                else:
                    st.success("No critical gaps Found! Great Job")
            # Part4
            st.divider()
            
            #Raw text
            with st.expander("See Extracted Resume Text"):
                st.caption("This is the raw text our AI extracted from your PDF. If this looks wrong, try a different PDF format.")
                st.text(resume_text)

            #Download Report
            report_text = f"""
            RESUME ANALYSIS REPORT
            ----------------------
            Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}
            Overall Score: {final_score}%

            SCORES:
            - Context Match: {sbert}%
            - Skill Match: {keyword_score}%
            - Vocabulary: {tfidf}%
            
            MISSING SKILLS:
            {', '.join(missing) if missing else 'None'}
            
            AI FEEDBACK:
            {chr(10).join(['- ' + i for i in feedback])}
            """
            st.download_button(
                label="Download the Full Report",
                data=report_text,
                file_name="Resume_Analysis_Report.txt",
                mime="text/plain"
            )
    elif not uploaded_file and job_description:
        st.error("Please ensure that the resume is uploaded!")
    elif uploaded_file and not job_description:
        st.error("Please ensure that the Job description is provided!")
    else:
        st.info(" Waiting for input.Please upload the documents in the column.")