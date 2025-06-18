# Configuration pour le nettoyage de données
# Modifiez ces paramètres selon vos besoins

class CleaningConfig:
    # Chemins des fichiers
    INPUT_PATH = "data/all.txt"
    OUTPUT_PATH = "data/cleaned.txt"
    
    # Paramètres de déduplication
    DUPLICATE_SIMILARITY_THRESHOLD = 0.9  # 0.8-0.95 recommandé
    
    # Filtres de longueur
    MIN_PHRASE_LENGTH = 10
    MAX_PHRASE_LENGTH = 1000
    
    # Performance
    BATCH_SIZE = 1000  # Augmenter si vous avez beaucoup de RAM
    NUM_WORKERS = None  # None = auto-détection, ou spécifiez un nombre
    
    # Options de qualité
    USE_GRAMMAR_CHECK = False  # ATTENTION: très lent mais améliore la qualité
    LANG_DETECT_MIN_LENGTH = 20  # Longueur min pour détection langue fiable
    
    # Patterns à exclure (vous pouvez en ajouter)
    ADDITIONAL_FORBIDDEN_PATTERNS = [
        # Ajoutez vos patterns ici, exemple:
        # r"\bspam\b",  # Mot "spam"
        # r"\d{4}-\d{4}-\d{4}-\d{4}",  # Numéros de carte
    ]
    
    # Mots-clés à exclure
    FORBIDDEN_KEYWORDS = [
        # Ajoutez des mots-clés à exclure, exemple:
        # "publicité", "spam", "promotion"
    ]
    
    # Paramètres MinHash
    MINHASH_NUM_PERM = 128  # Plus élevé = plus précis mais plus lent
    SHINGLE_SIZE = 5  # Taille des n-grammes pour la comparaison
    
    # Logging
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
    PROGRESS_INTERVAL = 10000  # Afficher le progrès tous les X éléments
    
    # Filtres avancés
    MAX_REPEATED_CHARS = 5  # Maximum de caractères répétés (aaaaa)
    MAX_CONSECUTIVE_CAPS = 10  # Maximum de majuscules consécutives
    MIN_WORD_COUNT = 3  # Minimum de mots par phrase
    MAX_WORD_COUNT = 200  # Maximum de mots par phrase
    
    # Ratio de caractères spéciaux autorisé
    MAX_SPECIAL_CHAR_RATIO = 0.3  # 30% max de caractères spéciaux
    
    # Langues acceptées (pour extension future)
    ACCEPTED_LANGUAGES = ["fr"]  # Français uniquement pour l'instant
    
    @classmethod
    def get_num_workers(cls):
        """Retourne le nombre de workers à utiliser"""
        if cls.NUM_WORKERS is None:
            import multiprocessing as mp
            return max(1, mp.cpu_count() - 1)
        return cls.NUM_WORKERS
    
    @classmethod
    def validate_config(cls):
        """Valide la configuration"""
        assert 0 < cls.DUPLICATE_SIMILARITY_THRESHOLD <= 1, "Threshold doit être entre 0 et 1"
        assert cls.MIN_PHRASE_LENGTH > 0, "Longueur minimale doit être positive"
        assert cls.MAX_PHRASE_LENGTH > cls.MIN_PHRASE_LENGTH, "Longueur max > min"
        assert cls.BATCH_SIZE > 0, "Batch size doit être positif"
        assert cls.MINHASH_NUM_PERM > 0, "Num perm doit être positif"
        assert cls.SHINGLE_SIZE > 0, "Shingle size doit être positif"
        print("✓ Configuration validée")

# Profils prédéfinis
class FastConfig(CleaningConfig):
    """Configuration pour traitement rapide (qualité réduite)"""
    USE_GRAMMAR_CHECK = False
    BATCH_SIZE = 2000
    MINHASH_NUM_PERM = 64
    DUPLICATE_SIMILARITY_THRESHOLD = 0.85

class QualityConfig(CleaningConfig):
    """Configuration pour haute qualité (plus lent)"""
    USE_GRAMMAR_CHECK = True
    BATCH_SIZE = 500
    MINHASH_NUM_PERM = 256
    DUPLICATE_SIMILARITY_THRESHOLD = 0.95
    MIN_PHRASE_LENGTH = 15
    MAX_REPEATED_CHARS = 3
    MAX_CONSECUTIVE_CAPS = 5

class BalancedConfig(CleaningConfig):
    """Configuration équilibrée (recommandée)"""
    USE_GRAMMAR_CHECK = False
    BATCH_SIZE = 1000
    MINHASH_NUM_PERM = 128
    DUPLICATE_SIMILARITY_THRESHOLD = 0.9
    MIN_PHRASE_LENGTH = 12