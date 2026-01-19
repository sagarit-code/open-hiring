const API_URL = 'http://localhost:8000';

const searchBtn = document.getElementById('searchBtn');
const searchInput = document.getElementById('searchInput');
const resultsSection = document.getElementById('results');
const resultsGrid = document.getElementById('resultsGrid');
const loadingModal = document.getElementById('loadingModal');
const typingText = document.getElementById('typingText');

// Typing animation for "Real Developers"
const phrases = [
    'Real Developers',
    'Top Talent',
    'Code Masters',
    'Tech Leaders',
    'AI Engineers'
];
let phraseIndex = 0;
let charIndex = 0;
let isDeleting = false;
let typingSpeed = 150;

function typeEffect() {
    const currentPhrase = phrases[phraseIndex];
    
    if (isDeleting) {
        typingText.textContent = currentPhrase.substring(0, charIndex - 1);
        charIndex--;
        typingSpeed = 50;
    } else {
        typingText.textContent = currentPhrase.substring(0, charIndex + 1);
        charIndex++;
        typingSpeed = 150;
    }
    
    if (!isDeleting && charIndex === currentPhrase.length) {
        // Pause at end
        typingSpeed = 2000;
        isDeleting = true;
    } else if (isDeleting && charIndex === 0) {
        isDeleting = false;
        phraseIndex = (phraseIndex + 1) % phrases.length;
        typingSpeed = 500;
    }
    
    setTimeout(typeEffect, typingSpeed);
}

// Start typing animation on load
setTimeout(typeEffect, 1000);

// Search functionality
searchBtn.addEventListener('click', handleSearch);
searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        handleSearch();
    }
});

async function handleSearch() {
    const query = searchInput.value.trim();
    
    if (!query) {
        showNotification('Please enter a search query', 'warning');
        return;
    }

    try {
        // Show loading
        loadingModal.style.display = 'flex';
        searchBtn.disabled = true;
        searchBtn.querySelector('.btn-text').style.display = 'none';
        searchBtn.querySelector('.btn-loader').style.display = 'inline';

        // Call API
        const response = await fetch(`${API_URL}/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: query })
        });

        if (!response.ok) {
            throw new Error('Search failed');
        }

        const data = await response.json();
        displayResults(data.ranked_repos || []);

    } catch (error) {
        console.error('Error:', error);
        showNotification('Search failed. Please make sure the backend server is running on port 8000.', 'error');
    } finally {
        // Hide loading
        loadingModal.style.display = 'none';
        searchBtn.disabled = false;
        searchBtn.querySelector('.btn-text').style.display = 'inline';
        searchBtn.querySelector('.btn-loader').style.display = 'none';
    }
}

function displayResults(repos) {
    resultsGrid.innerHTML = '';
    
    if (repos.length === 0) {
        resultsGrid.innerHTML = `
            <div style="text-align: center; padding: 3rem; color: #666;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🔍</div>
                <h3>No repositories found</h3>
                <p>Try a different query or adjust your search terms.</p>
            </div>
        `;
    } else {
        repos.forEach((repo, index) => {
            const card = createRepoCard(repo);
            // Stagger animation
            card.style.animationDelay = `${index * 0.1}s`;
            resultsGrid.appendChild(card);
        });
    }
    
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function createRepoCard(repo) {
    const card = document.createElement('div');
    card.className = 'repo-card';
    
    const repoUrl = repo.url || repo.link || '#';
    const repoName = repoUrl.split('/').slice(-2).join('/');
    
    card.innerHTML = `
        <div class="repo-header">
            <a href="${repoUrl}" target="_blank" class="repo-name" rel="noopener noreferrer">
                📦 ${repoName}
            </a>
            <div class="repo-stats">
                <div class="stat" title="Stars">
                    ⭐ ${formatNumber(repo.stars)}
                </div>
                <div class="stat" title="Forks">
                    🔱 ${formatNumber(repo.forks)}
                </div>
            </div>
        </div>
        <div class="repo-meta">
            ${repo.language ? `<span>📝 ${repo.language}</span>` : ''}
            ${repo.pushed_at ? `<span>🕒 Updated: ${formatDate(repo.pushed_at)}</span>` : ''}
        </div>
    `;
    
    return card;
}

function formatNumber(num) {
    if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'k';
    }
    return num;
}

function formatDate(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays < 30) {
        return `${diffDays} days ago`;
    } else if (diffDays < 365) {
        const months = Math.floor(diffDays / 30);
        return `${months} month${months > 1 ? 's' : ''} ago`;
    } else {
        const years = Math.floor(diffDays / 365);
        return `${years} year${years > 1 ? 's' : ''} ago`;
    }
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'error' ? '#f44336' : type === 'warning' ? '#ff9800' : '#4CAF50'};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 10000;
        animation: slideIn 0.3s ease-out;
        max-width: 400px;
    `;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

// Add CSS for notification animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);