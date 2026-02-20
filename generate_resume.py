from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing, Line

def create_resume_pdf(filename):
    # Standard margins for balanced look (approx 0.5 inch)
    doc = SimpleDocTemplate(filename, pagesize=letter,
                            rightMargin=36, leftMargin=36,
                            topMargin=36, bottomMargin=36)

    styles = getSampleStyleSheet()
    # Header styles
    styles.add(ParagraphStyle(name='Name', parent=styles['Heading1'], alignment=TA_CENTER, fontSize=20, spaceAfter=6))
    styles.add(ParagraphStyle(name='Contact', parent=styles['Normal'], alignment=TA_CENTER, fontSize=10, textColor=colors.black, leading=12, spaceAfter=2))
    
    # Section Header - Distinct with spacing for line
    styles.add(ParagraphStyle(name='SectionHeader', parent=styles['Heading2'], fontSize=11, spaceBefore=10, spaceAfter=2,
                              borderColor=colors.black, borderWidth=0, borderPadding=0,
                              keepWithNext=True, textTransform='uppercase'))
    
    # Content styles
    styles.add(ParagraphStyle(name='JobTitle', parent=styles['Heading3'], fontSize=10.5, spaceBefore=4, spaceAfter=1))
    styles.add(ParagraphStyle(name='Company', parent=styles['Normal'], fontSize=10.5, fontName='Helvetica-Oblique', spaceBefore=0, spaceAfter=2))
    styles.add(ParagraphStyle(name='Date', parent=styles['Normal'], fontSize=9, alignment=TA_LEFT))
    
    # Update BodyStyle
    styles['BodyText'].fontSize = 10
    styles['BodyText'].spaceBefore = 1
    styles['BodyText'].spaceAfter = 2
    styles['BodyText'].leading = 13
    
    # Bullets
    bullet_style = ParagraphStyle(name='Bullet', parent=styles['BodyText'], bulletIndent=10, leftIndent=20, spaceBefore=1, spaceAfter=1)

    # Line Separator Helper
    def get_line():
        # Page width (612) - Margins (72) = 540. detailed width to match text area
        d = Drawing(540, 1) 
        d.add(Line(0, 0, 540, 0))
        return d

    story = []

    # Header
    story.append(Paragraph("AHISHMON AC", styles['Name']))
    
    story.append(Paragraph("Data Analyst | Python Developer | Machine Learning Enthusiast", styles['Contact']))
    story.append(Paragraph("Phone: +91 90253 34122 | Email: ahishmon2003@gmail.com", styles['Contact']))
    story.append(Paragraph("Location: Coimbatore, Tamil Nadu, India", styles['Contact']))
    
    links = "LinkedIn: <a href='https://www.linkedin.com/in/ahishmon-318971285'>ahishmon-318971285</a> | Portfolio: <a href='https://Ahish99.github.io/portfolio/'>Ahish99.github.io/portfolio</a>"
    story.append(Paragraph(links, styles['Contact']))
    
    story.append(Spacer(1, 4))
    story.append(get_line())
    story.append(Spacer(1, 8))

    # Summary
    story.append(Paragraph("SUMMARY", styles['SectionHeader']))
    story.append(get_line())
    story.append(Spacer(1, 4))
    summary_text = """Results-oriented Data Analyst and MCA student with expertise in <b>Python, SQL, and Data Visualization</b>. Proven ability to build end-to-end analytical solutions, from data cleaning and preprocessing to predictive modeling and dashboard creation. Passionate about leveraging data to drive business decisions."""
    story.append(Paragraph(summary_text, styles['BodyText']))
    
    # Technical Skills
    story.append(Paragraph("TECHNICAL SKILLS", styles['SectionHeader']))
    story.append(get_line())
    story.append(Spacer(1, 4))
    skills = [
        "<b>Programming Languages:</b> Python, R, SQL",
        "<b>Data Analysis:</b> Pandas, NumPy, Excel (Advanced), Power Query",
        "<b>Machine Learning:</b> Scikit-Learn (Regression, Random Forest, Gradient Boosting)",
        "<b>Visualization:</b> Tableau, Matplotlib, Seaborn, Power BI",
        "<b>Tools:</b> Git/GitHub, Jupyter Notebook, VS Code, ReportLab"
    ]
    for skill in skills:
         story.append(Paragraph("• " + skill, styles['BodyText']))

    # Education
    story.append(Paragraph("EDUCATION", styles['SectionHeader']))
    story.append(get_line())
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("<b>Master of Computer Applications (MCA)</b>", styles['JobTitle']))
    story.append(Paragraph("<i>CMS College of Science and Commerce</i> | Expected: April 2026", styles['BodyText']))
    
    story.append(Spacer(1, 3))

    story.append(Paragraph("<b>Bachelor of Science in Computer Science</b>", styles['JobTitle']))
    story.append(Paragraph("<i>CMS College of Science and Commerce</i> | Graduated: April 2024", styles['BodyText']))
    story.append(Paragraph("GPA: 74.04%", styles['BodyText']))

    # Project Experience (Renamed)
    story.append(Paragraph("PROJECT EXPERIENCE", styles['SectionHeader']))
    story.append(get_line())
    story.append(Spacer(1, 4))
    
    # Project 1
    story.append(Paragraph("<b>Supermarket Sales Analysis System (Desktop App)</b>", styles['JobTitle']))
    story.append(Paragraph("<i>Python, Tkinter, Machine Learning (Scikit-Learn), Pandas</i>", styles['Company']))
    project1_bullets = [
        "<b>Developed a full-stack desktop application</b> to analyze supermarket sales data with a modern Dark Mode UI.",
        "<b>Engineered preprocessing pipelines</b> handling missing values and Label Encoding to optimize data for ML.",
        "<b>Built predictive models</b> (Random Forest, Gradient Boosting) for Total Sales and Ratings with high accuracy.",
        "<b>Implemented EDA tools</b> generating dynamic correlation heatmaps and distribution charts.",
        "<b>Designed automated reporting</b> using ReportLab to generate PDF summaries for stakeholders."
    ]
    for b in project1_bullets:
        story.append(Paragraph("• " + b, bullet_style))

    # Certifications
    story.append(Paragraph("CERTIFICATIONS", styles['SectionHeader']))
    story.append(get_line())
    story.append(Spacer(1, 4))
    
    # Combined AI Cert with marks
    story.append(Paragraph("<b>AI Driven Data Analyst</b> - TN Skill Corporation & NCVET (Feb 2026)", styles['Bullet']))
    marks_text = "<i>(Grade A+ 94%) Competencies: SQL (95), AI Models (94), Data Cleaning (93), Visualization (93)</i>"
    story.append(Paragraph(marks_text, ParagraphStyle(name='SubBullet', parent=styles['BodyText'], leftIndent=25, fontSize=9.5)))
        
    other_certs = [
        "<b>Data Base Management System</b> - NPTEL (Sep 2025)",
        "<b>Google Professional Data Analytics Certificate</b> (Feb 2025)",
        "<b>Validate Data in Google Sheets</b> (Jan 2025)",
        "<b>Google AI Essentials</b> (Dec 2024)"
    ]
    for c in other_certs:
        story.append(Paragraph("• " + c, bullet_style))

    doc.build(story)
    print(f"PDF generated: {filename}")

if __name__ == "__main__":
    create_resume_pdf("Ahishmon_Resume_Final_ATS.pdf")
