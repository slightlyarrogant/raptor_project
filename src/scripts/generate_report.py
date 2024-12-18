"""Generate an interactive HTML analysis report for a Raptor index."""

import logging
from pathlib import Path
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from typing import Dict, List, Optional
import click
from src.utils.config import DEFAULT_CONFIG
from src.utils.openai_client import UnifiedAIClient
import time
import shutil
from jinja2 import Environment, FileSystemLoader
import numpy as np

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# HTML template for the report
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <meta charset="utf-8">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .section {
            margin-bottom: 40px;
            padding: 20px;
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .visualization {
            margin: 20px 0;
            padding: 15px;
            border: 1px solid #eee;
            border-radius: 5px;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            display: inline-block;
            width: calc(33% - 20px);
            margin-right: 20px;
            vertical-align: top;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-label {
            color: #7f8c8d;
            font-size: 14px;
        }
        .recommendation {
            background-color: #e8f4f8;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 10px 0;
        }
        .tab {
            overflow: hidden;
            border: 1px solid #ccc;
            background-color: #f1f1f1;
            border-radius: 5px 5px 0 0;
        }
        .tab button {
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 14px 16px;
            transition: 0.3s;
        }
        .tab button:hover {
            background-color: #ddd;
        }
        .tab button.active {
            background-color: #3498db;
            color: white;
        }
        .tabcontent {
            display: none;
            padding: 20px;
            border: 1px solid #ccc;
            border-top: none;
            border-radius: 0 0 5px 5px;
        }
        .visible {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        <p>Generated: {{ timestamp }}</p>
        
        <div class="tab">
            <button class="tablinks active" onclick="openTab(event, 'summary')">Summary</button>
            <button class="tablinks" onclick="openTab(event, 'documents')">Documents</button>
            <button class="tablinks" onclick="openTab(event, 'tree')">Tree Analysis</button>
        </div>

        <div id="summary" class="tabcontent visible">
            <div class="section">
                <h2>Tree Overview</h2>
                <div class="metric-cards">
                    {% for metric in summary_metrics %}
                    <div class="metric-card">
                        <div class="metric-value">{{ metric.value }}</div>
                        <div class="metric-label">{{ metric.label }}</div>
                    </div>
                    {% endfor %}
                </div>
                {{ summary_content | safe }}
            </div>
        </div>

        <div id="documents" class="tabcontent">
            <div class="section">
                <h2>Document Analysis</h2>
                <div class="visualization">
                    <div id="lengthDist"></div>
                </div>
                <div class="visualization">
                    <div id="wordCloud"></div>
                </div>
                {{ documents_content | safe }}
            </div>
        </div>

        <div id="tree" class="tabcontent">
            <div class="section">
                <h2>Tree Structure Analysis</h2>
                <div class="visualization">
                    <div id="treeViz"></div>
                </div>
                <div class="visualization">
                    <div id="clusterDist"></div>
                </div>
                {{ tree_content | safe }}
            </div>
        </div>
    </div>

    <script>
        function openTab(evt, tabName) {
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
            }
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }

        // Plotly charts
        {{ plotly_charts | safe }}
    </script>
</body>
</html>
"""

class InteractiveReportGenerator:
    """Generate interactive HTML analysis reports for Raptor indices."""
    
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.data_dir = Path(f"data/{index_name}")
        self.analysis_dir = Path(f"analysis_outputs/{index_name}")
        self.report_dir = Path(f"reports/{index_name}")
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.ai_client = UnifiedAIClient()
        self.language = DEFAULT_CONFIG['summarization']['language']
        
    def create_plotly_charts(self, tree_stats: Dict, doc_analysis: Dict) -> str:
        """Create interactive Plotly charts for the report."""
        charts = []
        
        # Length distribution
        if 'lengths' in doc_analysis:
            fig = go.Figure(data=[go.Histogram(x=doc_analysis['lengths'])])
            fig.update_layout(
                title="Document Length Distribution",
                xaxis_title="Length (characters)",
                yaxis_title="Count"
            )
            charts.append(f"Plotly.newPlot('lengthDist', {fig.to_json()});")
        
        # Cluster distribution
        if 'clusters' in tree_stats:
            fig = go.Figure(data=[go.Bar(
                x=[c['name'] for c in tree_stats['clusters']],
                y=[c['size'] for c in tree_stats['clusters']]
            )])
            fig.update_layout(
                title="Cluster Size Distribution",
                xaxis_title="Cluster",
                yaxis_title="Size"
            )
            charts.append(f"Plotly.newPlot('clusterDist', {fig.to_json()});")
        
        # Tree visualization
        if 'tree_structure' in tree_stats:
            fig = go.Figure(data=[go.Treemap(
                labels=[node['name'] for node in tree_stats['tree_structure']],
                parents=[node.get('parent', '') for node in tree_stats['tree_structure']],
                values=[node.get('size', 1) for node in tree_stats['tree_structure']]
            )])
            fig.update_layout(title="Document Tree Structure")
            charts.append(f"Plotly.newPlot('treeViz', {fig.to_json()});")
        
        return "\n".join(charts)
        
    def generate_summary_metrics(self, tree_stats: Dict) -> List[Dict]:
        """Generate summary metrics for the report."""
        return [
            {
                'value': tree_stats.get('total_nodes', 0),
                'label': 'Total Nodes'
            },
            {
                'value': tree_stats.get('leaf_nodes', 0),
                'label': 'Leaf Nodes'
            },
            {
                'value': f"{tree_stats.get('avg_coherence', 0):.2f}",
                'label': 'Average Coherence'
            },
            {
                'value': tree_stats.get('max_depth', 0),
                'label': 'Maximum Depth'
            },
            {
                'value': f"{tree_stats.get('balance_score', 0):.2f}",
                'label': 'Balance Score'
            },
            {
                'value': len(tree_stats.get('clusters', [])),
                'label': 'Total Clusters'
            }
        ]
        
    def generate_report(self):
        """Generate the interactive HTML report."""
        try:
            logger.info(f"Generating interactive report for index: {self.index_name}")
            
            # Load data
            tree_stats_path = self.analysis_dir / "tree_viz/tree_stats.json"
            doc_analysis_path = self.analysis_dir / "document_analysis/document_analysis.json"
            
            with open(tree_stats_path, 'r') as f:
                tree_stats = json.load(f)
            with open(doc_analysis_path, 'r') as f:
                doc_analysis = json.load(f)
            
            # Generate content sections using AI
            summary_prompt = self.create_summary_prompt(tree_stats)
            documents_prompt = self.create_documents_prompt(doc_analysis)
            tree_prompt = self.create_tree_prompt(tree_stats)
            
            summary_content = self.ai_client.chat_completion(summary_prompt).choices[0].message.content
            documents_content = self.ai_client.chat_completion(documents_prompt).choices[0].message.content
            tree_content = self.ai_client.chat_completion(tree_prompt).choices[0].message.content
            
            # Create Plotly charts
            plotly_charts = self.create_plotly_charts(tree_stats, doc_analysis)
            
            # Generate summary metrics
            summary_metrics = self.generate_summary_metrics(tree_stats)
            
            # Prepare template data
            template_data = {
                'title': f"Raptor Analysis Report - {self.index_name}",
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'summary_metrics': summary_metrics,
                'summary_content': summary_content,
                'documents_content': documents_content,
                'tree_content': tree_content,
                'plotly_charts': plotly_charts
            }
            
            # Render template
            env = Environment(loader=FileSystemLoader('.'))
            template = env.from_string(HTML_TEMPLATE)
            html_content = template.render(**template_data)
            
            # Save report
            report_path = self.report_dir / f"analysis_report_{time.strftime('%Y%m%d_%H%M%S')}.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Copy static assets
            assets_dir = self.report_dir / "assets"
            assets_dir.mkdir(exist_ok=True)
            
            # Copy visualizations
            for viz_file in self.analysis_dir.rglob('*.png'):
                shutil.copy2(viz_file, assets_dir / viz_file.name)
            
            logger.info(f"Report generated successfully: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Failed to generate report: {str(e)}")
            raise
            
    def create_summary_prompt(self, tree_stats: Dict) -> List[Dict]:
        """Create prompt for summary section."""
        return [
            {"role": "system", "content": "You are a technical documentation expert specializing in document tree analysis."},
            {"role": "user", "content": f"""Generate a summary section for a document tree analysis report in {self.language} language.
Use the following statistics to create a detailed analysis with HTML formatting:

{json.dumps(tree_stats, indent=2)}

Include:
1. Overview of tree structure
2. Key metrics interpretation
3. Quality assessment
4. Main areas for improvement
5. Specific recommendations

Format the response in HTML with proper tags and classes. Use the 'recommendation' class for recommendations."""}
        ]
        
    def create_documents_prompt(self, doc_analysis: Dict) -> List[Dict]:
        """Create prompt for documents section."""
        return [
            {"role": "system", "content": "You are a technical documentation expert specializing in document analysis."},
            {"role": "user", "content": f"""Generate a documents section for an analysis report in {self.language} language.
Analyze the following data and create a detailed report with HTML formatting:

{json.dumps(doc_analysis, indent=2)}

Include:
1. Overview of document collection
2. Analysis of characteristics
3. Distribution patterns
4. Quality assessment
5. Recommendations for improvement

Format the response in HTML with proper tags and classes. Use the 'recommendation' class for recommendations."""}
        ]
        
    def create_tree_prompt(self, tree_stats: Dict) -> List[Dict]:
        """Create prompt for tree visualization section."""
        return [
            {"role": "system", "content": "You are a technical documentation expert specializing in tree structure analysis."},
            {"role": "user", "content": f"""Generate a tree visualization section for an analysis report in {self.language} language.
Analyze the following data and create a detailed report with HTML formatting:

{json.dumps(tree_stats, indent=2)}

Include:
1. Analysis of tree structure
2. Cluster quality assessment
3. Balance and depth analysis
4. Metric interpretation
5. Recommendations for improvement

Format the response in HTML with proper tags and classes. Use the 'recommendation' class for recommendations."""}
        ]

@click.command()
@click.option('--index-name', required=True, help='Name of the Raptor index to analyze')
def main(index_name: str):
    """Generate an interactive HTML analysis report for a Raptor index."""
    try:
        generator = InteractiveReportGenerator(index_name)
        report_path = generator.generate_report()
        print(f"\nInteractive report generated successfully: {report_path}")
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise

if __name__ == "__main__":
    main()
