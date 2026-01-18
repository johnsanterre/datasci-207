/**
 * DATASCI 207: Applied Machine Learning
 * Quiz Auto-Grading System
 * 
 * Features:
 * - Multiple choice with single correct answer
 * - Immediate feedback with explanations
 * - Shows why correct answer is right
 * - Shows why selected wrong answer is wrong
 * - Score tracking (session only)
 * - Reset functionality
 */

(function() {
    'use strict';

    // Quiz state
    let answered = {};
    let score = 0;
    let totalQuestions = 0;

    /**
     * Initialize the quiz system
     * Call this after DOM is loaded
     */
    function initQuiz() {
        const quizContainer = document.querySelector('.quiz-container');
        if (!quizContainer) return;

        const questions = quizContainer.querySelectorAll('.quiz-question');
        totalQuestions = questions.length;

        questions.forEach((question, index) => {
            setupQuestion(question, index);
        });

        // Setup action buttons
        const checkBtn = document.getElementById('check-answers');
        const resetBtn = document.getElementById('reset-quiz');

        if (checkBtn) {
            checkBtn.addEventListener('click', checkAllAnswers);
        }

        if (resetBtn) {
            resetBtn.addEventListener('click', resetQuiz);
        }

        updateScoreDisplay();
    }

    /**
     * Setup event listeners for a single question
     */
    function setupQuestion(questionEl, questionIndex) {
        const options = questionEl.querySelectorAll('input[type="radio"]');
        
        options.forEach(option => {
            option.addEventListener('change', function() {
                // If already answered, don't allow changes
                if (answered[questionIndex]) {
                    this.checked = false;
                    return;
                }

                handleAnswer(questionEl, questionIndex, this);
            });
        });
    }

    /**
     * Handle when user selects an answer
     */
    function handleAnswer(questionEl, questionIndex, selectedInput) {
        const correctAnswer = questionEl.dataset.correct;
        const selectedValue = selectedInput.value;
        const isCorrect = selectedValue === correctAnswer;

        // Mark as answered
        answered[questionIndex] = true;

        // Update score
        if (isCorrect) {
            score++;
        }

        // Disable all options for this question
        const allOptions = questionEl.querySelectorAll('input[type="radio"]');
        allOptions.forEach(opt => {
            opt.disabled = true;
        });

        // Show visual feedback on options
        const allLabels = questionEl.querySelectorAll('.quiz-options label');
        allLabels.forEach(label => {
            const input = label.querySelector('input[type="radio"]');
            if (input.value === correctAnswer) {
                label.classList.add('correct');
            } else if (input.value === selectedValue && !isCorrect) {
                label.classList.add('incorrect');
            }
        });

        // Show explanation feedback
        showFeedback(questionEl, isCorrect, selectedValue);

        // Update score display
        updateScoreDisplay();
    }

    /**
     * Show feedback for the answered question
     */
    function showFeedback(questionEl, isCorrect, selectedValue) {
        const feedbackEl = questionEl.querySelector('.quiz-feedback');
        if (!feedbackEl) return;

        // Get explanation data
        const explanations = JSON.parse(questionEl.dataset.explanations || '{}');
        const correctAnswer = questionEl.dataset.correct;

        let feedbackHTML = '';

        if (isCorrect) {
            feedbackEl.classList.remove('incorrect');
            feedbackEl.classList.add('correct');
            feedbackHTML = `
                <strong>Correct!</strong>
                <p>${explanations[correctAnswer] || 'Well done!'}</p>
            `;
        } else {
            feedbackEl.classList.remove('correct');
            feedbackEl.classList.add('incorrect');
            feedbackHTML = `
                <strong>Incorrect</strong>
                <p><strong>Why your answer is wrong:</strong> ${explanations[selectedValue] || 'This is not the correct answer.'}</p>
                <p><strong>Correct answer:</strong> ${explanations[correctAnswer] || 'See the correct option highlighted above.'}</p>
            `;
        }

        feedbackEl.innerHTML = feedbackHTML;
        feedbackEl.classList.add('show');
    }

    /**
     * Check all answers (for users who want to submit all at once)
     */
    function checkAllAnswers() {
        const questions = document.querySelectorAll('.quiz-question');
        
        questions.forEach((question, index) => {
            if (answered[index]) return; // Skip already answered

            const selected = question.querySelector('input[type="radio"]:checked');
            if (selected) {
                handleAnswer(question, index, selected);
            }
        });
    }

    /**
     * Reset the entire quiz
     */
    function resetQuiz() {
        answered = {};
        score = 0;

        const questions = document.querySelectorAll('.quiz-question');
        
        questions.forEach(question => {
            // Reset radio buttons
            const options = question.querySelectorAll('input[type="radio"]');
            options.forEach(opt => {
                opt.checked = false;
                opt.disabled = false;
            });

            // Reset label styles
            const labels = question.querySelectorAll('.quiz-options label');
            labels.forEach(label => {
                label.classList.remove('correct', 'incorrect');
            });

            // Hide feedback
            const feedback = question.querySelector('.quiz-feedback');
            if (feedback) {
                feedback.classList.remove('show', 'correct', 'incorrect');
                feedback.innerHTML = '';
            }
        });

        updateScoreDisplay();
    }

    /**
     * Update the score display
     */
    function updateScoreDisplay() {
        const scoreEl = document.getElementById('quiz-score');
        if (!scoreEl) return;

        const answeredCount = Object.keys(answered).length;
        
        if (answeredCount === 0) {
            scoreEl.innerHTML = `
                <span class="score-label">Questions</span>
                <span class="score-value">${totalQuestions}</span>
            `;
        } else {
            const percentage = Math.round((score / answeredCount) * 100);
            scoreEl.innerHTML = `
                <span class="score-label">Score</span>
                <span class="score-value">${score}/${answeredCount}</span>
                <span class="score-label">(${percentage}%)</span>
            `;
        }
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initQuiz);
    } else {
        initQuiz();
    }

    // Expose reset function globally if needed
    window.resetQuiz = resetQuiz;

})();
