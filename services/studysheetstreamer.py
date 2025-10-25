# Fixed Study Sheet Streamer - HTTP Quality + WebSocket Speed
# Key changes:
# 1. Generate COMPLETE sections (no word-by-word)
# 2. Use collapsible structure from HTTP version
# 3. Stream progress updates, not content chunks

import re
import json
import asyncio
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI

class StudySheetStreamer:
    """Handles progressive HTML streaming for study sheets - FIXED VERSION"""
    
    def __init__(self, session):
        self.session = session
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    async def stream_study_sheet_generation(self, topic: str, websocket, chat_id: str):
        """Main entry point for study sheet streaming"""
        try:
            language = self.session.user_language
            messages = self.get_status_messages(language)
            
            # Step 1: Analyze documents
            await self.send_status(websocket, "study_sheet_analyzing", messages["analyzing"])
            context = await self.get_document_context(topic)
            
            # Step 2: Generate dynamic outline  
            await self.send_status(websocket, "study_sheet_planning", messages["planning"])
            sections = await self.generate_dynamic_outline(topic, context, language)
            
            # Step 3: Send plan to frontend
            plan_steps = self.create_plan_steps(sections, language, messages)
            await websocket.send_text(json.dumps({
                "type": "stream_chunk",
                "data": {
                    "status": "study_sheet_plan_ready",
                    "topic": topic,
                    "language": language,
                    "steps": plan_steps,
                    "sections": sections
                }
            }))
            
            # Step 4: Send HTML skeleton (COLLAPSIBLE STRUCTURE - like HTTP version)
            skeleton_html = self.create_collapsible_skeleton(topic, sections, language)
            await websocket.send_text(json.dumps({
                "type": "stream_chunk",
                "data": {
                    "status": "study_sheet_html_skeleton",
                    "html_content": skeleton_html
                }
            }))
            
            # Step 5: Generate sections (COMPLETE, not word-by-word)
            await self.generate_complete_sections(
                topic, sections, context, language, skeleton_html, websocket
            )
            
        except Exception as e:
            await self.handle_error(websocket, str(e), language)
    
    async def generate_complete_sections(
    self, 
    topic: str, 
    sections: List[Dict], 
    context: str, 
    language: str, 
    base_html: str, 
    websocket
):
        """Generate ALL sections in PARALLEL using asyncio.gather - MUCH FASTER!"""
        
        section_weight = 70 / len(sections)
        current_progress = 15  # After planning
        
        # Step 1: Send "section_start" notifications for all sections
        print(f"🚀 Starting PARALLEL generation of {len(sections)} sections")
        
        # Step 2: Create all generation tasks AT ONCE
        generation_tasks = []
        for section in sections:
            # Notify that this section is starting
            await websocket.send_text(json.dumps({
                "type": "stream_chunk",
                "data": {
                    "status": "study_sheet_section_start",
                    "section_id": section["id"],
                    "section_title": section["title"],
                    "message": section["message"],
                    "progress": current_progress
                }
            }))
            
            # Create task (doesn't await yet - just schedules it)
            task = self.generate_rich_section_html(
                section, topic, context, language
            )
            generation_tasks.append((section, task))
        
        # Step 3: GENERATE ALL SECTIONS IN PARALLEL! 🔥
        print(f"⚡ Generating {len(sections)} sections in parallel...")
        
        # Wait for ALL sections to complete
        results = await asyncio.gather(
            *[task for _, task in generation_tasks],
            return_exceptions=True  # Don't fail if one section errors
        )
        
        print(f"✅ All {len(sections)} sections generated!")
        
        # Step 4: Update HTML with all completed sections
        current_html = base_html
        
        for i, ((section, _), section_html) in enumerate(zip(generation_tasks, results)):
            # Handle errors
            if isinstance(section_html, Exception):
                print(f"❌ Error generating section {section['id']}: {section_html}")
                section_html = f"<p>Error generating content for {section['title']}</p>"
            
            # Replace placeholder with generated content
            placeholder = f"{{{{CONTENT_{section['id']}}}}}"
            current_html = current_html.replace(placeholder, section_html)
            
            # Update badge from loading → complete
            current_html = self.update_section_badge(current_html, section["id"])
            
            # Calculate progress
            current_progress = 15 + (section_weight * (i + 1))
            
            # Send update for THIS section
            await websocket.send_text(json.dumps({
                "type": "stream_chunk",
                "data": {
                    "status": "study_sheet_content_update",
                    "html_content": current_html,
                    "progress": min(current_progress, 99)
                }
            }))
            
            # Notify section complete
            await websocket.send_text(json.dumps({
                "type": "stream_chunk",
                "data": {
                    "status": "study_sheet_section_complete", 
                    "section_id": section["id"],
                    "progress": current_progress
                }
            }))
            
            print(f"✅ Section {i+1}/{len(sections)} sent to client: {section['title']}")
        
        # Step 5: Send final completion
        await websocket.send_text(json.dumps({
            "type": "stream_chunk",
            "data": {
                "status": "study_sheet_complete",
                "html_content": current_html,
                "progress": 100
            }
        }))
        
        print(f"🎉 Study sheet generation complete!")
    
    async def generate_rich_section_html(
        self, 
        section: Dict, 
        topic: str, 
        context: str, 
        language: str
    ) -> str:
        """Generate RICH, COLOR-CODED section HTML"""
        
        prompts = {
            "english": f"""
            Generate comprehensive HTML content for the "{section['title']}" section 
            of a nursing study sheet about "{topic}".
            
            Context from documents: {context[:4000]}
            
            CRITICAL: Use COLOR-CODED CARDS for different information types:
            
            🔑 <div class="card card-blue"> - Key Concepts & Definitions
            ✅ <div class="card card-green"> - Clinical Applications & Nursing Actions
            ⚠️ <div class="card card-yellow"> - Warnings & Critical Alerts
            🧬 <div class="card card-purple"> - Pathophysiology & Disease Process
            🔍 <div class="card card-orange"> - Assessment Findings & Observations
            📚 <div class="card card-pink"> - Patient Education
            🚨 <div class="card card-red"> - Emergency & Critical Situations
            💊 <div class="card card-teal"> - Medications & Pharmacology
            
            Each card automatically gets an emoji from CSS. Structure:
            <div class="card card-COLOR">
            <div class="card-title">Title Here</div>
            <p>Content here...</p>
            <ul>
                <li>Point 1</li>
                <li>Point 2</li>
            </ul>
            </div>
            
            SPECIAL ELEMENTS:
            
            Highlights:
            - <span class="highlight">important term</span> (yellow)
            - <span class="highlight-red">danger sign</span> (red)
            - <span class="highlight-blue">key concept</span> (blue)
            - <span class="highlight-green">positive</span> (green)
            
            Lab Values:
            <span class="lab-value">
            <span class="value">7.35-7.45</span>
            <span class="unit">pH</span>
            </span>
            
            Dosages:
            <span class="dosage">2.5mg via nebulizer q4h PRN</span>
            
            Stats:
            <span class="stat-box stat-normal">SpO2: 95-100%</span>
            <span class="stat-box stat-abnormal">SpO2: <85%</span>
            
            Priority Levels:
            <div class="priority-high">🚨 High Priority: Respiratory distress</div>
            <div class="priority-medium">⚠️ Medium: Monitor vitals</div>
            <div class="priority-low">ℹ️ Low: Lifestyle education</div>
            
            Mnemonics (for assessment tools like SAMPLE, ABC):
            <div class="mnemonic">
            <div class="mnemonic-title">SAMPLE Assessment</div>
            <div class="mnemonic-letters">
                <div class="mnemonic-letter">
                <strong>S</strong> Symptoms
                </div>
                <div class="mnemonic-letter">
                <strong>A</strong> Allergies
                </div>
                <div class="mnemonic-letter">
                <strong>M</strong> Medications
                </div>
                <div class="mnemonic-letter">
                <strong>P</strong> Past Medical History
                </div>
                <div class="mnemonic-letter">
                <strong>L</strong> Last Oral Intake
                </div>
                <div class="mnemonic-letter">
                <strong>E</strong> Events Leading
                </div>
            </div>
            </div>
            
            STRUCTURE TEMPLATE:
            
            <p>Brief introduction to the concept...</p>
            
            <div class="card card-purple">
            <div class="card-title">Pathophysiology</div>
            <p>Explain disease process...</p>
            </div>
            
            <div class="card card-orange">
            <div class="card-title">Assessment Findings</div>
            <ul>
                <li><span class="highlight-red">Primary symptom</span>: Description</li>
                <li><span class="highlight">Key finding</span>: Description</li>
                <li>Normal range: <span class="lab-value"><span class="value">120-160</span> <span class="unit">mEq/L</span></span></li>
            </ul>
            </div>
            
            <div class="card card-yellow">
            <div class="card-title">Critical Alert</div>
            <p>Important warning...</p>
            </div>
            
            <div class="card card-green">
            <div class="card-title">Nursing Interventions</div>
            <ul>
                <li>Action 1</li>
                <li>Action 2</li>
            </ul>
            </div>
            
            <div class="card card-teal">
            <div class="card-title">Medications</div>
            <p><strong>Drug Name</strong>: <span class="dosage">dose and route</span></p>
            <ul>
                <li>Indication</li>
                <li>Side effects to monitor</li>
            </ul>
            </div>
            
            <div class="card card-pink">
            <div class="card-title">Patient Education</div>
            <p>Teaching points...</p>
            </div>
            
            Make it comprehensive (300-500 words), clinically accurate, and visually engaging!
            Use multiple card types per section.
            Return ONLY HTML - no markdown, no explanations, no code blocks.
            """,
            
            "french": f"""
            Générez un contenu HTML complet pour la section "{section['title']}" 
            d'une fiche d'étude infirmière sur "{topic}".
            
            Contexte des documents: {context[:4000]}
            
            CRITIQUE: Utilisez des CARTES COLORÉES pour différents types d'informations:
            
            🔑 <div class="card card-blue"> - Concepts Clés & Définitions
            ✅ <div class="card card-green"> - Applications Cliniques & Interventions Infirmières
            ⚠️ <div class="card card-yellow"> - Avertissements & Alertes Critiques
            🧬 <div class="card card-purple"> - Physiopathologie & Processus de la Maladie
            🔍 <div class="card card-orange"> - Résultats d'Évaluation & Observations
            📚 <div class="card card-pink"> - Éducation du Patient
            🚨 <div class="card card-red"> - Situations d'Urgence & Critiques
            💊 <div class="card card-teal"> - Médicaments & Pharmacologie
            
            Chaque carte obtient automatiquement un emoji du CSS. Structure:
            <div class="card card-COULEUR">
            <div class="card-title">Titre Ici</div>
            <p>Contenu ici...</p>
            <ul>
                <li>Point 1</li>
                <li>Point 2</li>
            </ul>
            </div>
            
            ÉLÉMENTS SPÉCIAUX:
            
            Surlignages:
            - <span class="highlight">terme important</span> (jaune)
            - <span class="highlight-red">signe de danger</span> (rouge)
            - <span class="highlight-blue">concept clé</span> (bleu)
            - <span class="highlight-green">positif</span> (vert)
            
            Valeurs de Laboratoire:
            <span class="lab-value">
            <span class="value">7,35-7,45</span>
            <span class="unit">pH</span>
            </span>
            
            Dosages:
            <span class="dosage">2,5mg par nébuliseur q4h PRN</span>
            
            Statistiques:
            <span class="stat-box stat-normal">SpO2: 95-100%</span>
            <span class="stat-box stat-abnormal">SpO2: <85%</span>
            
            Niveaux de Priorité:
            <div class="priority-high">🚨 Priorité Élevée: Détresse respiratoire</div>
            <div class="priority-medium">⚠️ Priorité Moyenne: Surveiller les signes vitaux</div>
            <div class="priority-low">ℹ️ Priorité Basse: Éducation sur le mode de vie</div>
            
            Mnémoniques (pour les outils d'évaluation comme SAMPLE, ABC):
            <div class="mnemonic">
            <div class="mnemonic-title">Évaluation SAMPLE</div>
            <div class="mnemonic-letters">
                <div class="mnemonic-letter">
                <strong>S</strong> Symptômes
                </div>
                <div class="mnemonic-letter">
                <strong>A</strong> Allergies
                </div>
                <div class="mnemonic-letter">
                <strong>M</strong> Médicaments
                </div>
                <div class="mnemonic-letter">
                <strong>P</strong> Passé Médical
                </div>
                <div class="mnemonic-letter">
                <strong>L</strong> Dernière Prise Orale
                </div>
                <div class="mnemonic-letter">
                <strong>E</strong> Événements Précédents
                </div>
            </div>
            </div>
            
            MODÈLE DE STRUCTURE:
            
            <p>Brève introduction au concept...</p>
            
            <div class="card card-purple">
            <div class="card-title">Physiopathologie</div>
            <p>Expliquer le processus de la maladie...</p>
            </div>
            
            <div class="card card-orange">
            <div class="card-title">Manifestations Cliniques</div>
            <ul>
                <li><span class="highlight-red">Symptôme principal</span>: Description</li>
                <li><span class="highlight">Résultat clé</span>: Description</li>
                <li>Plage normale: <span class="lab-value"><span class="value">120-160</span> <span class="unit">mEq/L</span></span></li>
            </ul>
            </div>
            
            <div class="card card-yellow">
            <div class="card-title">Alerte Critique</div>
            <p>Avertissement important...</p>
            </div>
            
            <div class="card card-green">
            <div class="card-title">Interventions Infirmières</div>
            <ul>
                <li>Action 1</li>
                <li>Action 2</li>
            </ul>
            </div>
            
            <div class="card card-teal">
            <div class="card-title">Médicaments</div>
            <p><strong>Nom du Médicament</strong>: <span class="dosage">dose et voie</span></p>
            <ul>
                <li>Indication</li>
                <li>Effets secondaires à surveiller</li>
            </ul>
            </div>
            
            <div class="card card-pink">
            <div class="card-title">Éducation du Patient</div>
            <p>Points d'enseignement...</p>
            </div>
            
            Rendez-le complet (300-500 mots), cliniquement précis et visuellement attrayant!
            Utilisez plusieurs types de cartes par section.
            Retournez UNIQUEMENT du HTML - pas de markdown, pas d'explications, pas de blocs de code.
            """
        }
        
        prompt = prompts.get(language, prompts["english"])
        
        try:
            response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            content = response.content.strip()
            
            # Clean any markdown code blocks if LLM ignores instructions
            if content.startswith("```html"):
                content = content.split("```html")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            return content
            
        except Exception as e:
            print(f"Error generating section: {e}")
            error_messages = {
                "english": f"<p>Error generating content for {section['title']}</p>",
                "french": f"<p>Erreur lors de la génération du contenu pour {section['title']}</p>"
            }
            return error_messages.get(language, error_messages["english"])
    
   
    def create_collapsible_skeleton(
    self, 
    topic: str, 
    sections: List[Dict], 
    language: str
) -> str:
        """Create collapsible section skeleton with enhanced color-coded design"""
    
        # Generate section HTML blocks
        sections_html = []
        for section in sections:
            section_html = f"""
    <div class="section" id="section-{section['id']}">
    <div class="section-header" onclick="toggleSection(this)">
        <div class="section-title">
        <span>{section['title']}</span>
        </div>
        <div class="section-badges">
        <span class="badge badge-loading" id="badge-{section['id']}"></span>
        <span class="chevron">▼</span>
        </div>
    </div>
    <div class="section-content">
        <div id="content-{section['id']}">
        {{{{CONTENT_{section['id']}}}}}
        </div>
    </div>
    </div>"""
            sections_html.append(section_html)
        
        sections_combined = "\n".join(sections_html)
        
        return f"""<!DOCTYPE html>
    <html lang="{language[:2]}">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{topic}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{ 
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        padding: 8px;
        min-height: 100vh;
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
        }}
        
        @media (min-width: 768px) {{
        body {{ padding: 20px; }}
        }}
        
        .container {{ 
        max-width: 1000px; 
        margin: 0 auto; 
        }}
        
        /* ============================================
        HEADER
        ============================================ */
        .header {{ 
        background: white; 
        padding: 16px;
        border-radius: 12px; 
        margin-bottom: 12px; 
        box-shadow: 0 4px 20px rgba(103, 126, 234, 0.15);
        border-top: 4px solid #667eea;
        }}
        
        @media (min-width: 768px) {{
        .header {{ 
            padding: 30px; 
            margin-bottom: 20px;
            border-radius: 16px;
        }}
        }}
        
        .header h1 {{ 
        color: #667eea; 
        font-size: 1.5rem;
        margin-bottom: 4px;
        word-wrap: break-word;
        font-weight: 800;
        }}
        
        @media (min-width: 768px) {{
        .header h1 {{ font-size: 2.2rem; }}
        }}
        
        .header p {{
        color: #64748b;
        font-size: 0.9rem;
        }}
        
        /* ============================================
        SECTIONS
        ============================================ */
        .section {{ 
        background: white; 
        margin: 8px 0;
        border-radius: 8px; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); 
        overflow: hidden;
        transition: box-shadow 0.3s ease;
        }}
        
        .section:hover {{
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
        }}
        
        @media (min-width: 768px) {{
        .section {{ 
            margin: 15px 0; 
            border-radius: 12px; 
        }}
        }}
        
        .section-header {{ 
        padding: 12px;
        cursor: pointer; 
        display: flex; 
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); 
        border-bottom: 2px solid #e5e7eb;
        gap: 8px;
        transition: background 0.2s ease;
        }}
        
        @media (min-width: 768px) {{
        .section-header {{ padding: 20px; }}
        }}
        
        .section-header:hover {{ 
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%); 
        }}
        
        .section-title {{ 
        font-size: 1rem;
        font-weight: 700; 
        color: #1e293b;
        flex: 1;
        word-wrap: break-word;
        line-height: 1.3;
        }}
        
        @media (min-width: 768px) {{
        .section-title {{ font-size: 1.2rem; }}
        }}
        
        .section-badges {{
        display: flex;
        align-items: center;
        gap: 8px;
        }}
        
        .section-content {{ 
        padding: 12px;
        display: none;
        overflow-x: auto;
        }}
        
        @media (min-width: 768px) {{
        .section-content {{ padding: 24px; }}
        }}
        
        .section.open .section-content {{ display: block; }}
        
        /* ============================================
        BADGES
        ============================================ */
        .badge {{ 
        padding: 4px 8px;
        border-radius: 12px; 
        font-size: 0.75rem;
        font-weight: 600; 
        display: inline-flex; 
        align-items: center; 
        gap: 4px;
        white-space: nowrap;
        flex-shrink: 0;
        transition: all 0.3s ease;
        }}
        
        @media (min-width: 768px) {{
        .badge {{ 
            padding: 6px 12px; 
            font-size: 0.85rem; 
            gap: 6px;
        }}
        }}
        
        .badge-loading {{ 
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
        color: #92400e;
        }}
        
        .badge-loading::before {{
        content: "🔄";
        animation: spin 2s linear infinite;
        display: inline-block;
        }}
        
        .badge-loaded {{ 
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); 
        color: #166534;
        }}
        
        .badge-loaded::before {{
        content: "✅";
        }}
        
        @keyframes spin {{ 
        to {{ transform: rotate(360deg); }} 
        }}
        
        .chevron {{ 
        transition: transform 0.3s ease;
        font-size: 1rem;
        color: #667eea;
        }}
        
        .section.open .chevron {{ transform: rotate(180deg); }}
        
        /* ============================================
        COLOR-CODED CARD SYSTEM
        ============================================ */
        .card {{
        padding: 16px;
        margin: 16px 0;
        border-radius: 12px;
        border-left: 4px solid;
        background: white;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
        word-wrap: break-word;
        }}
        
        @media (min-width: 768px) {{
        .card {{
            padding: 20px;
            margin: 20px 0;
        }}
        }}
        
        .card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        }}
        
        .card-title {{
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
        }}
        
        @media (min-width: 768px) {{
        .card-title {{ font-size: 1.15rem; }}
        }}
        
        /* Blue - Key Concepts & Definitions */
        .card-blue {{
        background: linear-gradient(135deg, #e3f2fd 0%, #f5f9ff 100%);
        border-left-color: #2196F3;
        }}
        
        .card-blue .card-title {{
        color: #1565C0;
        }}
        
        .card-blue .card-title::before {{
        content: "🔑";
        font-size: 1.2rem;
        }}
        
        /* Green - Clinical Applications & Nursing Actions */
        .card-green {{
        background: linear-gradient(135deg, #e8f5e9 0%, #f5fff6 100%);
        border-left-color: #4CAF50;
        }}
        
        .card-green .card-title {{
        color: #2E7D32;
        }}
        
        .card-green .card-title::before {{
        content: "✅";
        font-size: 1.2rem;
        }}
        
        /* Yellow - Warnings & Critical Alerts */
        .card-yellow {{
        background: linear-gradient(135deg, #fff9e6 0%, #fffef5 100%);
        border-left-color: #FFC107;
        }}
        
        .card-yellow .card-title {{
        color: #F57C00;
        }}
        
        .card-yellow .card-title::before {{
        content: "⚠️";
        font-size: 1.2rem;
        }}
        
        /* Purple - Pathophysiology & Disease Process */
        .card-purple {{
        background: linear-gradient(135deg, #f3e5f5 0%, #faf5fc 100%);
        border-left-color: #9C27B0;
        }}
        
        .card-purple .card-title {{
        color: #6A1B9A;
        }}
        
        .card-purple .card-title::before {{
        content: "🧬";
        font-size: 1.2rem;
        }}
        
        /* Orange - Assessment & Observations */
        .card-orange {{
        background: linear-gradient(135deg, #fff3e0 0%, #fffaf5 100%);
        border-left-color: #FF9800;
        }}
        
        .card-orange .card-title {{
        color: #E65100;
        }}
        
        .card-orange .card-title::before {{
        content: "🔍";
        font-size: 1.2rem;
        }}
        
        /* Pink - Patient Education */
        .card-pink {{
        background: linear-gradient(135deg, #fce4ec 0%, #fff5f8 100%);
        border-left-color: #E91E63;
        }}
        
        .card-pink .card-title {{
        color: #C2185B;
        }}
        
        .card-pink .card-title::before {{
        content: "📚";
        font-size: 1.2rem;
        }}
        
        /* Red - Emergency & Critical */
        .card-red {{
        background: linear-gradient(135deg, #ffebee 0%, #fff5f5 100%);
        border-left-color: #f44336;
        }}
        
        .card-red .card-title {{
        color: #c62828;
        }}
        
        .card-red .card-title::before {{
        content: "🚨";
        font-size: 1.2rem;
        }}
        
        /* Teal - Medications & Pharmacology */
        .card-teal {{
        background: linear-gradient(135deg, #e0f2f1 0%, #f5fffe 100%);
        border-left-color: #009688;
        }}
        
        .card-teal .card-title {{
        color: #00695C;
        }}
        
        .card-teal .card-title::before {{
        content: "💊";
        font-size: 1.2rem;
        }}
        
        /* ============================================
        INLINE HIGHLIGHTS
        ============================================ */
        .highlight {{
        background: linear-gradient(120deg, #ffd54f 0%, #ffeb3b 100%);
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        color: #f57f17;
        }}
        
        .highlight-blue {{
        background: linear-gradient(120deg, #bbdefb 0%, #e3f2fd 100%);
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        color: #1565C0;
        }}
        
        .highlight-red {{
        background: linear-gradient(120deg, #ffcdd2 0%, #ffebee 100%);
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        color: #c62828;
        }}
        
        .highlight-green {{
        background: linear-gradient(120deg, #c8e6c9 0%, #e8f5e9 100%);
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        color: #2E7D32;
        }}
        
        /* ============================================
        STAT BOXES (for numbers/data)
        ============================================ */
        .stat-box {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        background: white;
        border: 2px solid;
        border-radius: 8px;
        font-weight: 700;
        margin: 4px;
        font-size: 0.9rem;
        }}
        
        .stat-normal {{
        border-color: #4CAF50;
        color: #2E7D32;
        background: #e8f5e9;
        }}
        
        .stat-abnormal {{
        border-color: #f44336;
        color: #c62828;
        background: #ffebee;
        }}
        
        .stat-warning {{
        border-color: #FF9800;
        color: #E65100;
        background: #fff3e0;
        }}
        
        /* ============================================
        NURSING MNEMONICS
        ============================================ */
        .mnemonic {{
        background: linear-gradient(135deg, #f3e5f5 0%, #faf5fc 100%);
        border: 2px solid #9C27B0;
        border-radius: 12px;
        padding: 16px;
        margin: 20px 0;
        }}
        
        .mnemonic-title {{
        font-size: 1.3rem;
        font-weight: 700;
        color: #6A1B9A;
        margin-bottom: 12px;
        text-align: center;
        }}
        
        .mnemonic-letters {{
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        justify-content: center;
        }}
        
        .mnemonic-letter {{
        background: white;
        border: 2px solid #9C27B0;
        border-radius: 8px;
        padding: 12px;
        min-width: 150px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        }}
        
        .mnemonic-letter strong {{
        font-size: 1.5rem;
        color: #9C27B0;
        display: block;
        margin-bottom: 4px;
        }}
        
        /* ============================================
        PRIORITY LEVELS
        ============================================ */
        .priority-high {{
        background: #ffebee;
        border-left: 4px solid #f44336;
        padding: 12px;
        margin: 8px 0;
        border-radius: 8px;
        font-weight: 600;
        color: #c62828;
        }}
        
        .priority-medium {{
        background: #fff3e0;
        border-left: 4px solid #FF9800;
        padding: 12px;
        margin: 8px 0;
        border-radius: 8px;
        font-weight: 600;
        color: #E65100;
        }}
        
        .priority-low {{
        background: #e8f5e9;
        border-left: 4px solid #4CAF50;
        padding: 12px;
        margin: 8px 0;
        border-radius: 8px;
        font-weight: 600;
        color: #2E7D32;
        }}
        
        /* ============================================
        DOSAGE & LAB VALUES
        ============================================ */
        .dosage {{
        background: #e0f2f1;
        border: 2px solid #009688;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px;
        font-family: 'Courier New', monospace;
        font-weight: 700;
        color: #00695C;
        display: inline-block;
        font-size: 0.95rem;
        }}
        
        .lab-value {{
        background: white;
        border: 2px solid #2196F3;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-weight: 600;
        }}
        
        .lab-value .value {{
        font-size: 1.2rem;
        color: #1565C0;
        }}
        
        .lab-value .unit {{
        font-size: 0.9rem;
        color: #64B5F6;
        }}
        
        /* ============================================
        TABLES
        ============================================ */
        table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin: 16px 0;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}
        
        thead {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        }}
        
        th {{
        padding: 12px;
        text-align: left;
        font-weight: 600;
        font-size: 0.95rem;
        }}
        
        tbody tr {{
        background: white;
        transition: background 0.2s;
        }}
        
        tbody tr:nth-child(even) {{
        background: #f8f9fa;
        }}
        
        tbody tr:hover {{
        background: #f3e5f5;
        }}
        
        td {{
        padding: 12px;
        border-bottom: 1px solid #e0e0e0;
        font-size: 0.9rem;
        }}
        
        /* ============================================
        TYPOGRAPHY
        ============================================ */
        h3 {{ 
        color: #667eea; 
        margin: 20px 0 12px;
        font-size: 1.15rem;
        word-wrap: break-word;
        font-weight: 700;
        }}
        
        @media (min-width: 768px) {{
        h3 {{ 
            margin: 24px 0 14px; 
            font-size: 1.35rem; 
        }}
        }}
        
        p {{ 
        margin: 10px 0;
        line-height: 1.7; 
        color: #334155;
        word-wrap: break-word;
        font-size: 0.95rem;
        }}
        
        @media (min-width: 768px) {{
        p {{ 
            margin: 12px 0; 
            line-height: 1.75;
            font-size: 1rem;
        }}
        }}
        
        ul {{ 
        margin: 12px 0 12px 20px;
        }}
        
        @media (min-width: 768px) {{
        ul {{ margin: 14px 0 14px 24px; }}
        }}
        
        li {{ 
        margin: 8px 0;
        color: #475569;
        line-height: 1.6;
        word-wrap: break-word;
        font-size: 0.95rem;
        }}
        
        @media (min-width: 768px) {{
        li {{ 
            margin: 10px 0; 
            line-height: 1.7;
            font-size: 1rem;
        }}
        }}
        
        strong {{ 
        color: #1e293b; 
        font-weight: 700; 
        }}
        
        img {{ 
        max-width: 100%; 
        height: auto;
        border-radius: 8px;
        margin: 12px 0;
        }}
        
        .section-content * {{
        max-width: 100%;
        overflow-wrap: break-word;
        }}
    </style>
    </head>
    <body>
    <div class="container">
        <div class="header">
        <h1>📚 {topic}</h1>
        <p>Comprehensive Nursing Study Guide</p>
        </div>
        {sections_combined}
    </div>
    <script>
        function toggleSection(header) {{
        header.parentElement.classList.toggle('open');
        }}
        // Auto-open first section
        document.querySelector('.section')?.classList.add('open');
    </script>
    </body>
    </html>"""
    
    def update_section_badge(self, html: str, section_id: str) -> str:
        """Update section badge from loading to complete"""
        new_badge = f'<span class="badge badge-loaded" id="badge-{section_id}"></span>'
        
        # Replace the entire badge span
        pattern = f'<span class="badge badge-loading" id="badge-{section_id}">.*?</span>'
        html = re.sub(pattern, new_badge, html, flags=re.DOTALL)
        
        return html
    
    async def generate_dynamic_outline(self, topic: str, context: str, language: str) -> List[Dict]:
        """Generate content-based outline in target language"""
        
        prompts = {
            "english": f"""
            Analyze this document content about "{topic}" and create 5-6 main sections 
            for a comprehensive nursing study sheet. Focus on what's actually covered in the documents.
            
            Document content: {context[:3000]}
            
            Return as JSON array with this exact format:
            [
              {{"id": "overview", "title": "Overview and Introduction", "message": "Building overview section..."}},
              {{"id": "pathophysiology", "title": "Pathophysiology and Disease Process", "message": "Analyzing disease mechanisms..."}},
              {{"id": "assessment", "title": "Clinical Assessment", "message": "Compiling assessment criteria..."}},
              {{"id": "interventions", "title": "Nursing Interventions", "message": "Detailing nursing care..."}},
              {{"id": "management", "title": "Patient Management", "message": "Building management strategies..."}},
              {{"id": "education", "title": "Patient Education", "message": "Creating education guidelines..."}}
            ]
            
            Make section titles specific to the content (e.g., "Cardiac Assessment" not just "Assessment").
            Focus on nursing-relevant sections: pathophysiology, assessment, interventions, management, education.
            """,
            
            "french": f"""
            Analysez ce contenu de document sur "{topic}" et créez 5-6 sections principales 
            pour une fiche d'étude infirmière complète. Concentrez-vous sur ce qui est réellement couvert.
            
            Contenu du document: {context[:3000]}
            
            Retournez en format JSON:
            [
              {{"id": "apercu", "title": "Aperçu et Introduction", "message": "Construction de l'aperçu..."}},
              {{"id": "physiopathologie", "title": "Physiopathologie", "message": "Analyse des mécanismes..."}},
              {{"id": "evaluation", "title": "Évaluation Clinique", "message": "Compilation des critères..."}},
              {{"id": "interventions", "title": "Interventions Infirmières", "message": "Détails des soins..."}},
              {{"id": "gestion", "title": "Gestion du Patient", "message": "Stratégies de gestion..."}},
              {{"id": "education", "title": "Éducation du Patient", "message": "Lignes directrices..."}}
            ]
            
            Rendez les titres spécifiques au contenu.
            """
        }
        
        prompt = prompts.get(language, prompts["english"])
        
        try:
            response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            content = response.content.strip()
            
            # Clean JSON if wrapped in markdown
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0]
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0]
            
            sections = json.loads(content)
            return sections
            
        except Exception as e:
            print(f"Error generating outline: {e}")
            return self.get_fallback_sections(language)
    
    async def get_document_context(self, topic: str) -> str:
        """Get document context using existing search tool"""
        from tools.quiztools import search_documents
        
        try:
            search_result = await search_documents.ainvoke({
                "query": topic,
                "max_results": 10  # Increased from 8
            })
            return search_result.get("context", "")
        except Exception as e:
            print(f"Error getting document context: {e}")
            return ""
    
    async def send_status(self, websocket, status: str, message: str):
        """Send status update"""
        await websocket.send_text(json.dumps({
            "type": "stream_chunk",
            "data": {
                "status": status,
                "message": message
            }
        }))
    
    def get_status_messages(self, language: str) -> Dict[str, str]:
        """Get localized status messages"""
        messages = {
            "english": {
                "analyzing": "Analyzing your documents...",
                "planning": "Creating study sheet outline...",
                "generating": "Generating content...",
                "completing": "Finalizing study sheet...",
                "retrying": "Retrying generation...",
                "complete": "Study sheet complete!"
            },
            "french": {
                "analyzing": "Analyse de vos documents...",
                "planning": "Création du plan d'étude...",
                "generating": "Génération du contenu...",
                "completing": "Finalisation de la fiche d'étude...",
                "retrying": "Nouvelle tentative de génération...",
                "complete": "Fiche d'étude terminée!"
            }
        }
        return messages.get(language, messages["english"])
    
    def create_plan_steps(self, sections: List[Dict], language: str, messages: Dict) -> List[Dict]:
        """Create plan steps for progress tracking"""
        base_weight = 15
        section_weight = 70 / len(sections)
        
        steps = [
            {"id": "planning", "title": "Planning" if language == "english" else "Planification", 
             "message": messages["planning"], "weight": base_weight}
        ]
        
        for section in sections:
            steps.append({
                "id": section["id"],
                "title": section["title"], 
                "message": section["message"],
                "weight": section_weight
            })
        
        steps.append({
            "id": "completion", "title": "Completion" if language == "english" else "Finalisation",
            "message": messages["completing"], "weight": base_weight
        })
        
        return steps
    
    def get_fallback_sections(self, language: str) -> List[Dict]:
        """Fallback sections if outline generation fails"""
        if language == "french":
            return [
                {"id": "apercu", "title": "Aperçu", "message": "Construction de l'aperçu..."},
                {"id": "physiopathologie", "title": "Physiopathologie", "message": "Analyse des mécanismes..."},
                {"id": "evaluation", "title": "Évaluation Clinique", "message": "Compilation des critères..."},
                {"id": "interventions", "title": "Interventions Infirmières", "message": "Détails des soins..."},
                {"id": "education", "title": "Éducation du Patient", "message": "Lignes directrices..."}
            ]
        else:
            return [
                {"id": "overview", "title": "Overview", "message": "Building overview section..."},
                {"id": "pathophysiology", "title": "Pathophysiology", "message": "Analyzing disease mechanisms..."},
                {"id": "assessment", "title": "Clinical Assessment", "message": "Compiling assessment criteria..."},
                {"id": "interventions", "title": "Nursing Interventions", "message": "Detailing nursing care..."},
                {"id": "education", "title": "Patient Education", "message": "Creating education guidelines..."}
            ]
    
    async def handle_error(self, websocket, error_message: str, language: str):
        """Handle errors with retry logic"""
        messages = self.get_status_messages(language)
        
        await websocket.send_text(json.dumps({
            "type": "stream_chunk",
            "data": {
                "status": "study_sheet_error",
                "message": f"{messages['retrying']} ({error_message})"
            }
        }))