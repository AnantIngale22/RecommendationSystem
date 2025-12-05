import markdown
import os

def convert_to_html():
    md_file = "consultant_report_v3.md"
    html_file = "consultant_report_v3.html"
    
    if not os.path.exists(md_file):
        print("❌ Markdown file not found.")
        return

    with open(md_file, "r") as f:
        text = f.read()
        
    # Convert Markdown to HTML
    html_content = markdown.markdown(text, extensions=['tables', 'fenced_code'])
    
    # Create Full HTML with Dark Theme CSS and Mermaid
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Keek Consultant Report v3</title>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ 
                startOnLoad: true, 
                theme: 'dark',
                themeVariables: {{ fontSize: '13px', fontFamily: 'arial' }}
            }});
        </script>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                background-color: #1e1e1e;
                color: #e0e0e0;
                line-height: 1.6;
                padding: 40px;
                max-width: 1000px;
                margin: 0 auto;
            }}
            h1, h2, h3 {{
                color: #ffffff;
                border-bottom: 1px solid #333;
                padding-bottom: 10px;
            }}
            code {{
                background-color: #2d2d2d;
                padding: 2px 5px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }}
            pre {{
                background-color: #2d2d2d;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #444;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #333;
            }}
            .mermaid {{
                background-color: #1e1e1e;
                padding: 20px;
                border-radius: 5px;
                text-align: center;
            }}
            @media print {{
                body {{
                    background-color: #1e1e1e !important;
                    -webkit-print-color-adjust: exact;
                    print-color-adjust: exact;
                    color: #e0e0e0 !important;
                }}
                .mermaid {{
                    background-color: #1e1e1e !important;
                    -webkit-print-color-adjust: exact;
                    print-color-adjust: exact;
                    page-break-inside: avoid;
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 0 auto;
                }}
                h1, h2, h3, th, td, code, pre {{
                    color: #e0e0e0 !important;
                    border-color: #444 !important;
                }}
                @page {{
                    size: A4;
                    margin: 10mm;
                }}
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Fix Mermaid Block (Markdown converts it to <pre><code>)
    # We need to replace <pre><code class="language-mermaid"> with <div class="mermaid">
    full_html = full_html.replace('<pre><code class="language-mermaid">', '<div class="mermaid">')
    full_html = full_html.replace('</code></pre>', '</div>')
    
    # Remove the init directive from the text if it shows up
    full_html = full_html.replace("%%{init: {'theme': 'dark', 'themeVariables': { 'fontSize': '20px', 'fontFamily': 'arial', 'darkMode': true }}}%%", "")

    with open(html_file, "w") as f:
        f.write(full_html)
        
    print(f"✅ Generated HTML Report: {html_file}")

if __name__ == "__main__":
    convert_to_html()
