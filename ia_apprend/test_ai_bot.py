import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    from . import ai_bot
    from .ai_bot import EntityExtractor, IntentClassifier, LearningBot
except ImportError:  # pragma: no cover - script fallback
    import ai_bot
    from ai_bot import EntityExtractor, IntentClassifier, LearningBot


class LearningBotTests(unittest.TestCase):
    def test_teach_and_answer_exact_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)
            bot.teach("Salut", "Bonjour !")

            self.assertEqual(bot.answer("salut"), "Bonjour !")

    def test_save_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)
            bot.teach("Quelle heure est-il", "Je ne connais pas l'heure.")
            bot.answer("reponds en francais")
            bot.remember_subject("python", "Q: test | R: test")
            bot.save()

            reloaded = LearningBot.load(memory_path)
            self.assertEqual(
                reloaded.answer("quelle heure est-il"),
                "Je ne connais pas l'heure.",
            )
            self.assertIn("langue", reloaded.preferences)
            self.assertIn("python", reloaded.subject_memory)

    def test_empty_message_gets_helpful_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            self.assertIn("Ecris une question", bot.answer("   "))

    def test_summary_mode_summarizes_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            self.assertIn("Envoie-moi le texte", bot.answer("résume ça"))
            summary = bot.answer(
                "Python est un langage simple. Il est utilisé pour l'automatisation. Il sert aussi à la science des données."
            )
            self.assertTrue(summary)
            self.assertNotIn("Envoie-moi le texte", summary)

    def test_lesson_mode_builds_revision_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            intro = bot.answer("mode apprentissage")
            answer = bot.answer(
                "Cours: La photosynthese permet aux plantes de fabriquer du sucre avec la lumiere. "
                "Les feuilles captent le dioxyde de carbone et l'eau. "
                "La chlorophylle aide a utiliser l'energie lumineuse. "
                "La plante libere de l'oxygene dans l'air."
            )

            self.assertIn("Mode apprentissage", intro)
            self.assertIn("lecon apprise", answer.lower())
            self.assertIn("Questions pour reviser", answer)
            self.assertGreaterEqual(len(bot.documents), 1)
            self.assertTrue(any("photosynthese" in item["question"] for item in bot.examples))

    def test_correction_mode_explains_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            self.assertIn("Envoie-moi la phrase", bot.answer("corrige ça"))
            answer = bot.answer("je suis aller au college")
            self.assertIn("Correction", answer)
            self.assertIn("Explication", answer)

    def test_question_mode_and_common_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            self.assertIn("Envoie-moi la question", bot.answer("j'ai une question"))
            answer = bot.answer("Pourquoi le ciel est bleu ?")
            self.assertTrue(answer)
            self.assertIn("bleu", answer.lower())
            self.assertIn("lumi", answer.lower())

    def test_direct_messages_are_useful(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            self.assertIn("question", bot.answer("bonjour").lower())
            self.assertIn("repondre", bot.answer("test").lower())
            self.assertIn("amelior", bot.answer("ameliore").lower())

    def test_common_app_questions_have_grounded_answers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            self.assertIn("mettre", bot.answer("a quoi sert Render ?").lower())
            self.assertIn("code", bot.answer("c'est quoi GitHub ?").lower())
            api_answer = bot.answer("c'est quoi une API ?").lower()
            self.assertIn("serveur", api_answer)
            self.assertNotIn("cle api", api_answer)

    def test_calculation_mode_solves_simple_expression(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            answer = bot.answer("calcule 2 + 2")

            self.assertIn("Calcul", answer)
            self.assertIn("4", answer)

    def test_calculation_mode_handles_parentheses_and_powers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            answer = bot.answer("12 * (3 + 4)")
            power = bot.answer("2 puissance 3")

            self.assertIn("84", answer)
            self.assertIn("8", power)

    def test_calculation_mode_after_question_followup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            self.assertIn("Envoie-moi la question", bot.answer("question"))
            answer = bot.answer("20 % de 50")

            self.assertIn("10", answer)
            self.assertIn("Calcul", answer)

    def test_synthetic_catalog_reaches_large_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            self.assertGreaterEqual(bot.synthetic_example_count(), 1_000_000_000)
            self.assertGreaterEqual(bot.total_example_count(), 1_000_000_000)

    def test_synthetic_catalog_answers_new_topics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            answer = bot.answer("Qu'est-ce qu'un navigateur ?")
            self.assertTrue("Réponse" in answer or "Reponse" in answer)
            self.assertIn("navigateur", answer.lower())

            explanation = bot.answer("Quelles sont les limites du web ?")
            self.assertTrue("Réponse" in explanation or "Reponse" in explanation)
            self.assertIn("web", explanation.lower())

    def test_ambiguous_question_gets_clarification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            answer = bot.answer("je veux savoir")
            self.assertIn("contexte", answer.lower())

    def test_explain_mode_explains_code(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            answer = bot.answer("explique ce code")
            self.assertIn("code", answer.lower())
            self.assertNotIn("Envoie-moi le texte", answer)
            self.assertTrue("réponse" in answer.lower() or "reponse" in answer.lower())

    def test_translate_mode_produces_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            self.assertIn("Envoie-moi le texte", bot.answer("traduis ce texte"))
            answer = bot.answer("hello")
            self.assertTrue(answer)

    def test_plan_mode_creates_outline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            self.assertIn("Envoie-moi le texte ou le sujet", bot.answer("fais un plan"))
            answer = bot.answer("L'IA doit comprendre, résumer et corriger.")
            self.assertIn("Plan", answer)

    def test_quiz_mode_asks_and_scores_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)
            bot.teach("Qu'est-ce que Python ?", "Python est un langage de programmation.")

            self.assertIn("Quel sujet", bot.answer("quiz"))
            question = bot.answer("python")
            self.assertIn("Question pour python", question)
            self.assertIn("qu'est-ce que python", question.lower())
            score = bot.answer("Python est un langage de programmation.")
            self.assertIn("Bonne reponse", score)

    def test_memory_captures_user_statements(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            bot.answer("je m'appelle Lucie")
            answer = bot.answer("que sais-tu de moi ?")

            self.assertIn("Lucie", answer)
            self.assertIn("Ce que je me rappelle", answer)

    def test_memory_sources_track_origin(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            bot.remember_note_from_source("user", "L'utilisateur s'appelle Lucie")
            bot.remember_note_from_source("document", "Document ajoute: Histoire")

            sources = bot.list_memory_sources()
            self.assertIn("user", sources)
            self.assertIn("document", sources)
            self.assertTrue(any("Lucie" in note for note in sources["user"]))

    def test_reference_followup_reuses_last_subject(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            opening = bot.answer("Parle-moi de celine dion")
            answer = bot.answer("et son mari ?")

            self.assertIn("parler de", opening.lower())
            self.assertIn("dion", opening.lower())
            self.assertTrue(
                "reste sur" in answer.lower()
                or "parles probablement de" in answer.lower()
                or "garde le contexte" in answer.lower()
            )
            self.assertIn("Sujet courant", bot.get_conversation_summary())

    def test_followup_expands_subject_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            bot.answer("Parle-moi de celine dion")
            answer = bot.answer("et son mari ?")

            self.assertIn("mari", answer.lower())
            self.assertTrue(
                "je n'ai pas" in answer.lower()
                or "je garde" in answer.lower()
                or "contexte" in answer.lower()
            )

    def test_topic_opening_sets_context_and_invites_follow_up(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            answer = bot.answer("Parle-moi de l'histoire")

            self.assertIn("histoire", answer.lower())
            self.assertTrue(
                "pose" in answer.lower()
                or "question" in answer.lower()
                or "suite" in answer.lower()
            )

    def test_question_uses_open_subject_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            bot.answer("Parle-moi de celine dion")
            answer = bot.answer("qui est-ce ?")

            self.assertTrue(
                "sujet" in answer.lower()
                or "contexte" in answer.lower()
                or "je pars du sujet" in answer.lower()
                or "je me base" in answer.lower()
            )

    def test_question_profile_handles_definition_style_questions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            answer = bot.answer("Qu'est-ce que la gravite ?")

            self.assertIn("grav", answer.lower())
            self.assertIn("je peux", answer.lower())

    def test_question_profile_handles_how_style_questions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            answer = bot.answer("Comment fonctionne un serveur ?")

            self.assertIn("serveur", answer.lower())
            self.assertIn("je peux", answer.lower())

    def test_example_request_without_subject_gets_helpful_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            answer = bot.answer("Donne un exemple de comparaison")

            self.assertIn("exemple", answer.lower())
            self.assertIn("compar", answer.lower())
            self.assertTrue("usage" in answer.lower() or "cas réel" in answer.lower())

    def test_style_request_changes_response_tone(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            answer = bot.answer("explique simplement le serveur")

            self.assertIn("serveur", answer.lower())
            self.assertTrue(
                "version simple" in answer.lower()
                or "idée clé" in answer.lower()
                or "réponse" in answer.lower()
                or "reponse" in answer.lower()
            )

    def test_unrelated_question_does_not_reuse_old_subject(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            bot.answer("Parle-moi de celine dion")
            answer = bot.answer("Qu'est-ce que la gravite ?")

            self.assertIn("grav", answer.lower())
            self.assertNotIn("celine", answer.lower())

    def test_unrelated_statement_does_not_force_old_subject(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            bot.answer("Parle-moi d'un serveur")
            answer = bot.answer("test interface")

            self.assertNotIn("serveur", answer.lower())

    def test_direct_smalltalk_answers_naturally(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            answer = bot.answer("bonjour")

            self.assertIn("Bonjour", answer)

    def test_conversation_summary_persists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)
            bot.teach("Qu'est-ce qu'un serveur ?", "Un ordinateur qui fournit un service.")
            bot.answer("je m'appelle Lucie")
            bot.answer("qu'est-ce qu'un serveur ?")
            bot.refresh_conversation_summary()
            bot.save()

            reloaded = LearningBot.load(memory_path)
            summary = reloaded.get_conversation_summary()
            self.assertTrue(summary)
            self.assertIn("Lucie", summary)
            self.assertIn("serveur", summary.lower())

    def test_subject_brief_persists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            bot.answer("Parle-moi de celine dion")
            bot.answer("et son mari ?")
            bot.save()

            reloaded = LearningBot.load(memory_path)
            briefs = reloaded.list_subject_briefs()
            self.assertTrue(any("dion" in key.lower() for key in briefs))
            self.assertTrue(any("Sujet:" in brief for brief in briefs.values()))

    def test_subject_memory_recall(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)
            bot.remember_subject("histoire", "Q: Napoléon | R: Empereur français")
            bot.save()

            reloaded = LearningBot.load(memory_path)
            answer = reloaded.answer("que sais tu sur histoire ?")
            self.assertIn("histoire", answer.lower())
            self.assertIn("Napoléon", answer)

    def test_document_memory_recall(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)
            bot.add_document(
                "Histoire",
                "Napoléon a été empereur. Il a marqué l'histoire de France.",
            )
            bot.save()

            reloaded = LearningBot.load(memory_path)
            answer = reloaded.answer("que dit le document sur napoleon ?")
            self.assertIn("Napoléon", answer)

    def test_unknown_question_gets_clarifying_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            answer = bot.answer("comment faire ?")

            self.assertIn("étapes", answer.lower())
            self.assertTrue("entrée" in answer.lower() or "traitement" in answer.lower())
            self.assertTrue("exemple" in answer.lower() or "par exemple" in answer.lower())

    def test_conversation_control_expands_previous_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            bot.answer("Qu'est-ce qu'un serveur ?")
            answer = bot.answer("explique plus")

            self.assertIn("developpe", answer.lower())
            self.assertIn("serveur", answer.lower())
            self.assertIn("exemple", answer.lower())

    def test_conversation_control_gives_example_from_previous_topic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            bot.answer("Qu'est-ce que Python ?")
            answer = bot.answer("donne un exemple")

            self.assertIn("exemple", answer.lower())
            self.assertIn("python", answer.lower())

    def test_corrupted_memory_falls_back_to_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            memory_path.write_text("{broken json", encoding="utf-8")

            bot = LearningBot.load(memory_path)

            self.assertIsNotNone(bot.startup_warning)
            self.assertGreaterEqual(len(bot.examples), 1)

    def test_intent_classifier_predicts_question(self) -> None:
        classifier = IntentClassifier.train(
            [
                {"text": "Quelle est la capitale de la France ?", "label": "question"},
                {"text": "Peux-tu corriger cette phrase ?", "label": "correction"},
                {"text": "Je veux apprendre quelque chose.", "label": "demande"},
            ]
        )

        prediction = classifier.predict("Où se trouve la Tour Eiffel ?")

        self.assertIsNotNone(prediction)
        assert prediction is not None
        self.assertEqual(prediction.label, "question")
        self.assertGreater(prediction.confidence, 0.0)

    def test_robot_command_detection_parses_move_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            command = bot._robot_action_from_message("robot avance 2 secondes vitesse 30")

            self.assertIsNotNone(command)
            assert command is not None
            self.assertEqual(command["action"], "forward")
            self.assertEqual(command["message"], "forward")
            self.assertEqual(command["speed"], 30)
            self.assertAlmostEqual(command["duration"], 2.0)

    def test_robot_command_uses_bridge_sender(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            with mock.patch.object(
                LearningBot,
                "_send_robot_command",
                return_value="Robot: forward.",
            ) as sender:
                answer = bot.answer("robot avance")

            sender.assert_called_once()
            self.assertIn("Robot", answer)
            self.assertIn("forward", answer.lower())

    def test_robot_diagnostic_request_is_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            command = bot._robot_action_from_message("test robot")

            self.assertIsNotNone(command)
            assert command is not None
            self.assertEqual(command["action"], "diagnostic")
            self.assertEqual(command["message"], "diagnostic")

    def test_robot_message_field_is_sent_to_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            with mock.patch.object(
                LearningBot,
                "_send_robot_command",
                return_value="Robot: forward.",
            ) as sender:
                bot.answer("robot avance")

            payload = sender.call_args.args[0]
            self.assertEqual(payload["message"], "forward")
            self.assertEqual(payload["action"], "forward")

    def test_robot_bridge_status_handles_missing_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            with mock.patch.object(ai_bot, "ROBOT_BRIDGE_URL", ""):
                status = bot.robot_bridge_status()

            self.assertFalse(status["ok"])
            self.assertFalse(status["connected"])
            self.assertEqual(status["command_count"], 0)

    def test_robot_bridge_status_reads_health_payload(self) -> None:
        class DummyResponse:
            def __init__(self, payload: dict[str, object]) -> None:
                self._payload = payload

            def __enter__(self) -> "DummyResponse":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(self._payload).encode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            bot = LearningBot.load(memory_path)

            payload = {
                "ok": True,
                "message": "ready",
                "connected": True,
                "serial_port": "COM9",
                "board_type": "mcore",
                "last_action": "forward",
                "last_payload": {"left": 35, "right": 35},
                "last_at": 12.5,
                "last_error": "",
                "command_count": 7,
                "uptime_seconds": 42.25,
            }

            with mock.patch.object(ai_bot, "ROBOT_BRIDGE_URL", "http://robot"), mock.patch.object(
                ai_bot,
                "urlopen",
                return_value=DummyResponse(payload),
            ):
                status = bot.robot_bridge_status()

            self.assertTrue(status["ok"])
            self.assertTrue(status["connected"])
            self.assertEqual(status["serial_port"], "COM9")
            self.assertEqual(status["board_type"], "mcore")
            self.assertEqual(status["command_count"], 7)
            self.assertAlmostEqual(status["uptime_seconds"], 42.25)

    def test_entity_extractor_finds_known_entities(self) -> None:
        extractor = EntityExtractor.train(
            [
                {
                    "text": "Jean va a Paris demain.",
                    "entities": [
                        {"text": "Jean", "label": "personne"},
                        {"text": "Paris", "label": "lieu"},
                        {"text": "demain", "label": "date"},
                    ],
                }
            ]
        )

        prediction = extractor.predict("Jean va a Paris demain.")

        self.assertIsNotNone(prediction)
        assert prediction is not None
        self.assertEqual(len(prediction.entities), 3)
        self.assertEqual(prediction.entities[0].label, "personne")

    def test_extra_question_pack_answers_network_question(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app_root = Path(temp_dir) / "lucie"
            app_root.mkdir()
            source = Path(ai_bot.__file__).resolve().parent.parent / "lucie_extra_questions.json"
            (app_root / "lucie_extra_questions.json").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            memory_path = app_root / "ia_apprend" / "memory.json"
            bot = LearningBot.load(memory_path)

            answer = bot.answer("a quoi sert une adresse ip")

            self.assertIn("reseau", answer.lower())
            self.assertTrue("identifier" in answer.lower() or "appareil" in answer.lower())

    def test_extra_question_pack_answers_robot_question(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app_root = Path(temp_dir) / "lucie"
            app_root.mkdir()
            source = Path(ai_bot.__file__).resolve().parent.parent / "lucie_extra_questions.json"
            (app_root / "lucie_extra_questions.json").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            memory_path = app_root / "ia_apprend" / "memory.json"
            bot = LearningBot.load(memory_path)

            answer = bot.answer("c'est quoi un pca9685")

            self.assertIn("servomoteurs", answer.lower())
            self.assertIn("raspberry", answer.lower())

    def test_extra_question_pack_answers_science_question(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app_root = Path(temp_dir) / "lucie"
            app_root.mkdir()
            source = Path(ai_bot.__file__).resolve().parent.parent / "lucie_extra_questions.json"
            (app_root / "lucie_extra_questions.json").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            memory_path = app_root / "ia_apprend" / "memory.json"
            bot = LearningBot.load(memory_path)

            answer = bot.answer("qui etait marie curie")

            self.assertIn("scientifique", answer.lower())
            self.assertTrue("radioactivite" in answer.lower() or "nobel" in answer.lower())

    def test_extra_question_pack_2_answers_without_generic_detour(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app_root = Path(temp_dir) / "lucie"
            app_root.mkdir()
            root = Path(ai_bot.__file__).resolve().parent.parent
            for name in ("lucie_extra_questions.json", "lucie_extra_questions_2.json"):
                (app_root / name).write_text((root / name).read_text(encoding="utf-8"), encoding="utf-8")
            memory_path = app_root / "ia_apprend" / "memory.json"
            bot = LearningBot.load(memory_path)

            checks = {
                "pourquoi localhost n'est pas visible sur google": "localhost",
                "comment rapprocher lucie de chatgpt": "modele",
                "pourquoi mon servo tremble": "servo",
                "c est quoi une hallucination d ia": "hallucination",
            }

            for question, expected in checks.items():
                with self.subTest(question=question):
                    answer = bot.answer(question).lower()
                    self.assertIn(expected, answer)
                    self.assertNotIn("une reponse en 'pourquoi'", answer)
                    self.assertNotIn("fonctionnement : chats", answer)


if __name__ == "__main__":
    unittest.main()
