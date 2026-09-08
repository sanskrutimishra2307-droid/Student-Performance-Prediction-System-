"""
Explainable AI (XAI) and Risk Factor Attribution Suite.
Provides global model interpretability, local student profile breakdown, and automated academic intervention recommendations.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

class ExplainabilityEngine:
    """
    Explainable AI & Decision Support Engine for Educational Predictions.
    """
    
    @staticmethod
    def analyze_student_risk_factors(student_dict: Dict[str, Any], prediction: str) -> Dict[str, Any]:
        """
        Analyzes individual student inputs and pinpoints actionable risk factors and positive assets.
        """
        risk_factors = []
        positive_assets = []
        recommendations = []
        
        # 1. Study Hours
        study_hours = float(student_dict.get("study_hours", 0))
        if study_hours < 2.0:
            risk_factors.append({
                "factor": "Critically Low Study Time",
                "detail": f"{study_hours} hrs/day is significantly below the recommended minimum of 3.5 hrs/day.",
                "severity": "HIGH"
            })
            recommendations.append("Establish a structured daily study schedule aiming for at least 3.0-4.0 hours with regular breaks.")
        elif study_hours < 3.5:
            risk_factors.append({
                "factor": "Sub-optimal Study Time",
                "detail": f"{study_hours} hrs/day may be insufficient for high grade attainment.",
                "severity": "MEDIUM"
            })
            recommendations.append("Gradually increase daily focused study time by 45-60 minutes.")
        else:
            positive_assets.append({
                "factor": "Strong Study Commitment",
                "detail": f"{study_hours} hrs/day represents robust academic dedication."
            })
            
        # 2. Attendance
        attendance = float(student_dict.get("attendance_percentage", 100))
        if attendance < 65.0:
            risk_factors.append({
                "factor": "Chronic Absenteeism",
                "detail": f"Attendance is at {attendance}%, well below the institutional 75% threshold.",
                "severity": "CRITICAL"
            })
            recommendations.append("Initiate immediate attendance counseling and weekly check-ins with academic advisor.")
        elif attendance < 75.0:
            risk_factors.append({
                "factor": "Moderate Attendance Deficit",
                "detail": f"Attendance is at {attendance}%.",
                "severity": "MEDIUM"
            })
            recommendations.append("Target 90%+ class attendance over the next 4 weeks to prevent concept gaps.")
        else:
            positive_assets.append({
                "factor": "High Attendance Reliability",
                "detail": f"{attendance}% attendance supports consistent concept mastery."
            })
            
        # 3. Subject scores (if available)
        for subj in ["math_score", "science_score", "english_score"]:
            if subj in student_dict and student_dict[subj] is not None:
                score = float(student_dict[subj])
                subj_name = subj.replace("_score", "").capitalize()
                if score < 45.0:
                    risk_factors.append({
                        "factor": f"Vulnerable Foundation in {subj_name}",
                        "detail": f"Score of {score}/100 requires urgent remedial tutoring.",
                        "severity": "HIGH"
                    })
                    recommendations.append(f"Enroll in peer-tutoring or weekly remedial workshops for {subj_name}.")
                elif score >= 80.0:
                    positive_assets.append({
                        "factor": f"Excellence in {subj_name}",
                        "detail": f"High score of {score}/100."
                    })
                    
        # 4. Travel Time
        travel_time = str(student_dict.get("travel_time", ""))
        if travel_time in [">60 min", "30-60 min"]:
            risk_factors.append({
                "factor": "Substantial Commute Fatigue",
                "detail": f"Travel time of '{travel_time}' reduces available evening study and rest time.",
                "severity": "LOW"
            })
            recommendations.append("Utilize travel time for audio lectures or flashcard review; explore hybrid study options.")
            
        # 5. Internet & Study Method
        internet = str(student_dict.get("internet_access", "")).lower()
        if internet in ["no", "0", "false"]:
            risk_factors.append({
                "factor": "Digital Resource Inaccessibility",
                "detail": "Lack of home internet restricts access to digital study materials and online coaching.",
                "severity": "MEDIUM"
            })
            recommendations.append("Provide access to campus computer labs, offline repository downloads, and library reserves.")
            
        study_method = str(student_dict.get("study_method", "")).lower()
        if study_method in ["coaching", "online videos"]:
            positive_assets.append({
                "factor": "Effective Modern Learning Method",
                "detail": f"Utilizing '{study_method}' has a high empirical correlation with distinction grades."
            })
            
        # Overall assessment summary
        if prediction.lower() in ["at-risk", "fail", "e", "f"]:
            urgency = "HIGH PRIORITY - IMMEDIATE INTERVENTION NEEDED"
            summary_statement = "Student exhibits significant vulnerability markers. Targeted academic support and attendance recovery plan recommended."
        elif prediction.lower() in ["pass", "c", "d"]:
            urgency = "MODERATE - PERFORMANCE OPTIMIZATION"
            summary_statement = "Student is in a stable passing band but can elevate to distinction with targeted study habit refinements."
        else:
            urgency = "LOW - ADVANCED ENRICHMENT"
            summary_statement = "Student demonstrates exemplary performance indicators. Encourage mentorship participation and advanced project work."
            
        return {
            "urgency": urgency,
            "summary_statement": summary_statement,
            "risk_factors": risk_factors,
            "positive_assets": positive_assets,
            "recommended_interventions": recommendations
        }
