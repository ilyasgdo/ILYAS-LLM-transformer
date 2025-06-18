import re
import multiprocessing as mp
from functools import partial
from datasketch import MinHash, MinHashLSH
import language_tool_python
from langdetect import detect, DetectorFactory
import time
from typing import List, Tuple, Optional, Dict
import logging
import json
import os
import re
from collections import Counter
from config_cleaning import CleaningConfig, FastConfig, QualityConfig, BalancedConfig

# Assurer la reproductibilité
DetectorFactory.seed = 0

class UltraOptimizedCleaner:
    def __init__(self, config=None):
        self.config = config or BalancedConfig()
        self.config.validate_config()
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.LOG_LEVEL),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Statistiques
        self.stats = {
            'total_lines': 0,
            'too_short': 0,
            'too_long': 0,
            'forbidden_patterns': 0,
            'wrong_language': 0,
            'grammar_errors': 0,
            'duplicates': 0,
            'quality_issues': 0,
            'final_kept': 0,
            'processing_time': 0
        }
        
        # Patterns compilés
        self.forbidden_patterns = self._compile_patterns()
        self.tag_re = re.compile(r'<[^>]+>')
        self.whitespace_re = re.compile(r'\s+')
        self.non_printable_re = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]')
        # Patterns de nettoyage spécifiques
        self.generic_phrase_re = re.compile(r'^Voici une phrase en français\s*:\s*', re.IGNORECASE)
        self.phrase_pattern_re = re.compile(r'La phrase est\s*:\s*', re.IGNORECASE)
        self.quotes_cleanup_re = re.compile(r'^["\'\u201c\u201d]+|["\'\u201c\u201d]+$')  # Guillemets en début/fin
        
        # Cache pour optimisation
        self.lang_cache = {}
        self.pattern_cache = {}
        
        # LanguageTool
        self.tool = None
        if self.config.USE_GRAMMAR_CHECK:
            self._init_language_tool()
    
    def _compile_patterns(self):
        """Compile tous les patterns interdits"""
        base_patterns = [
            re.compile(r"http\S+", re.IGNORECASE),
            re.compile(r"www\.\S+", re.IGNORECASE),
            re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
            re.compile(r"[^\w\s,.!?;:()\[\]{}\"'-]"),
            re.compile(r"\b\d{10,}\b"),
            re.compile(f"[A-Z]{{{self.config.MAX_CONSECUTIVE_CAPS},}}"),
            re.compile(f"(.)\\1{{{self.config.MAX_REPEATED_CHARS},}}"),
            re.compile(r"^Voici une phrase en français\s*:", re.IGNORECASE),  # Phrases génériques
            re.compile(r"La phrase est\s*:", re.IGNORECASE),  # Patterns de phrases
        ]
        
        # Ajouter patterns additionnels
        for pattern in self.config.ADDITIONAL_FORBIDDEN_PATTERNS:
            base_patterns.append(re.compile(pattern, re.IGNORECASE))
        
        return base_patterns
    
    def _init_language_tool(self):
        """Initialise LanguageTool avec gestion d'erreur"""
        try:
            self.tool = language_tool_python.LanguageTool('fr')
            self.logger.info("LanguageTool initialisé")
        except Exception as e:
            self.logger.warning(f"Impossible d'initialiser LanguageTool: {e}")
            self.tool = None
    
    def get_shingles(self, text: str, k: int = None) -> set:
        """Tokenise optimisé avec cache"""
        k = k or self.config.SHINGLE_SIZE
        text = self.whitespace_re.sub(' ', text.lower())
        
        if len(text) < k:
            return {text}
        
        return {text[i:i+k] for i in range(len(text) - k + 1)}
    
    def has_forbidden_patterns(self, text: str) -> bool:
        """Vérification optimisée avec cache"""
        cache_key = hash(text[:50])  # Cache basé sur début du texte
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]
        
        result = any(pattern.search(text) for pattern in self.forbidden_patterns)
        self.pattern_cache[cache_key] = result
        return result
    
    def has_forbidden_keywords(self, text: str) -> bool:
        """Vérification des mots-clés interdits"""
        if not self.config.FORBIDDEN_KEYWORDS:
            return False
        
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in self.config.FORBIDDEN_KEYWORDS)
    
    def check_quality_metrics(self, text: str) -> bool:
        """Vérifications de qualité avancées"""
        words = text.split()
        
        # Nombre de mots
        if len(words) < self.config.MIN_WORD_COUNT or len(words) > self.config.MAX_WORD_COUNT:
            return False
        
        # Ratio de caractères spéciaux
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if special_chars / len(text) > self.config.MAX_SPECIAL_CHAR_RATIO:
            return False
        
        # Vérifier la diversité des mots (éviter répétitions)
        if len(set(words)) / len(words) < 0.5:  # Moins de 50% de mots uniques
            return False
        
        return True
    
    def is_french_optimized(self, text: str) -> bool:
        """Détection de langue optimisée"""
        cache_key = ' '.join(text.split()[:3]).lower()
        if cache_key in self.lang_cache:
            return self.lang_cache[cache_key]
        
        if len(text) < self.config.LANG_DETECT_MIN_LENGTH:
            result = True
        else:
            try:
                lang = detect(text)
                result = lang in self.config.ACCEPTED_LANGUAGES
            except:
                result = True
        
        self.lang_cache[cache_key] = result
        return result
    
    def clean_text_advanced(self, text: str) -> Optional[str]:
        """Nettoyage avancé avec toutes les vérifications"""
        # Nettoyage basique
        text = self.tag_re.sub('', text)
        text = self.non_printable_re.sub('', text)
        
        # Suppression des patterns génériques
        text = self.generic_phrase_re.sub('', text)
        text = self.phrase_pattern_re.sub('', text)
        
        # Suppression des guillemets en début/fin
        text = self.quotes_cleanup_re.sub('', text)
        
        text = self.whitespace_re.sub(' ', text).strip()
        
        # Vérifications de longueur
        if len(text) < self.config.MIN_PHRASE_LENGTH:
            self.stats['too_short'] += 1
            return None
        
        if len(text) > self.config.MAX_PHRASE_LENGTH:
            self.stats['too_long'] += 1
            return None
        
        # Patterns interdits
        if self.has_forbidden_patterns(text):
            self.stats['forbidden_patterns'] += 1
            return None
        
        # Mots-clés interdits
        if self.has_forbidden_keywords(text):
            self.stats['forbidden_patterns'] += 1
            return None
        
        # Détection de langue
        if not self.is_french_optimized(text):
            self.stats['wrong_language'] += 1
            return None
        
        # Métriques de qualité
        if not self.check_quality_metrics(text):
            self.stats['quality_issues'] += 1
            return None
        
        return text
    
    def process_line(self, line: str) -> Optional[str]:
        """Traite une ligne complète"""
        self.stats['total_lines'] += 1
        
        cleaned = self.clean_text_advanced(line)
        if cleaned is None:
            return None
        
        # Correction grammaticale
        if self.tool and self.config.USE_GRAMMAR_CHECK:
            try:
                matches = self.tool.check(cleaned)
                if matches:
                    cleaned = language_tool_python.utils.correct(cleaned, matches)
                    self.stats['grammar_errors'] += len(matches)
            except Exception as e:
                self.logger.debug(f"Erreur correction: {e}")
        
        return cleaned
    
    def create_minhash(self, text: str) -> MinHash:
        """Crée un MinHash optimisé"""
        shingles = self.get_shingles(text)
        m = MinHash(num_perm=self.config.MINHASH_NUM_PERM)
        for shingle in shingles:
            m.update(shingle.encode('utf8'))
        return m
    
    def save_stats(self, filename: str = "cleaning_stats.json"):
        """Sauvegarde les statistiques"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Statistiques sauvegardées dans {filename}")
    
    def print_stats(self):
        """Affiche les statistiques détaillées"""
        total = self.stats['total_lines']
        if total == 0:
            return
        
        print("\n" + "="*50)
        print("STATISTIQUES DE NETTOYAGE")
        print("="*50)
        print(f"Total lignes traitées: {total:,}")
        print(f"Lignes gardées: {self.stats['final_kept']:,} ({self.stats['final_kept']/total*100:.2f}%)")
        print(f"Temps de traitement: {self.stats['processing_time']:.2f}s")
        print(f"Vitesse: {total/self.stats['processing_time']:.1f} lignes/sec")
        print("\nRaisons d'exclusion:")
        print(f"  - Trop courtes: {self.stats['too_short']:,} ({self.stats['too_short']/total*100:.2f}%)")
        print(f"  - Trop longues: {self.stats['too_long']:,} ({self.stats['too_long']/total*100:.2f}%)")
        print(f"  - Patterns interdits: {self.stats['forbidden_patterns']:,} ({self.stats['forbidden_patterns']/total*100:.2f}%)")
        print(f"  - Mauvaise langue: {self.stats['wrong_language']:,} ({self.stats['wrong_language']/total*100:.2f}%)")
        print(f"  - Problèmes qualité: {self.stats['quality_issues']:,} ({self.stats['quality_issues']/total*100:.2f}%)")
        print(f"  - Doublons: {self.stats['duplicates']:,} ({self.stats['duplicates']/total*100:.2f}%)")
        if self.stats['grammar_errors'] > 0:
            print(f"  - Erreurs corrigées: {self.stats['grammar_errors']:,}")
        print("="*50)

def process_batch_optimized(batch_lines: List[str], config) -> List[str]:
    """Traite un batch avec la configuration donnée"""
    cleaner = UltraOptimizedCleaner(config)
    results = []
    
    for line in batch_lines:
        cleaned = cleaner.process_line(line.strip())
        if cleaned:
            results.append(cleaned)
    
    return results, cleaner.stats

def main(config_type="balanced"):
    """Fonction principale avec choix de configuration"""
    # Sélection de la configuration
    configs = {
        "fast": FastConfig(),
        "quality": QualityConfig(),
        "balanced": BalancedConfig(),
        "custom": CleaningConfig()
    }
    
    config = configs.get(config_type, BalancedConfig())
    cleaner = UltraOptimizedCleaner(config)
    
    start_time = time.time()
    cleaner.logger.info(f"Démarrage avec configuration: {config_type}")
    
    # LSH pour déduplication
    lsh = MinHashLSH(
        threshold=config.DUPLICATE_SIMILARITY_THRESHOLD,
        num_perm=config.MINHASH_NUM_PERM
    )
    
    # Compteur global pour les clés uniques
    global_line_counter = 0
    
    with open(config.INPUT_PATH, 'r', encoding='utf-8') as fin, \
         open(config.OUTPUT_PATH, 'w', encoding='utf-8') as fout:
        
        batch = []
        all_stats = Counter()
        
        for line_num, raw_line in enumerate(fin, 1):
            batch.append(raw_line)
            
            if len(batch) >= config.BATCH_SIZE:
                # Traitement parallèle
                num_workers = config.get_num_workers()
                
                if num_workers > 1:
                    chunk_size = max(1, len(batch) // num_workers)
                    chunks = [batch[i:i+chunk_size] for i in range(0, len(batch), chunk_size)]
                    
                    with mp.Pool(num_workers) as pool:
                        func = partial(process_batch_optimized, config=config)
                        results = pool.map(func, chunks)
                    
                    # Combiner résultats et stats
                    cleaned_lines = []
                    for chunk_result, chunk_stats in results:
                        cleaned_lines.extend(chunk_result)
                        all_stats.update(chunk_stats)
                else:
                    cleaned_lines, batch_stats = process_batch_optimized(batch, config)
                    all_stats.update(batch_stats)
                
                # Déduplication
                for cleaned_line in cleaned_lines:
                    minhash = cleaner.create_minhash(cleaned_line)
                    
                    if not lsh.query(minhash):
                        lsh.insert(f"line_{global_line_counter}", minhash)
                        fout.write(cleaned_line + '\n')
                        cleaner.stats['final_kept'] += 1
                        global_line_counter += 1
                    else:
                        cleaner.stats['duplicates'] += 1
                
                # Mise à jour stats globales
                for key, value in all_stats.items():
                    cleaner.stats[key] = value
                
                # Progress
                if cleaner.stats['total_lines'] % config.PROGRESS_INTERVAL == 0:
                    elapsed = time.time() - start_time
                    rate = cleaner.stats['total_lines'] / elapsed
                    cleaner.logger.info(
                        f"Traité: {cleaner.stats['total_lines']:,}, "
                        f"Gardé: {cleaner.stats['final_kept']:,}, "
                        f"Taux: {rate:.1f} lignes/sec"
                    )
                
                batch = []
        
        # Dernier batch
        if batch:
            cleaned_lines, batch_stats = process_batch_optimized(batch, config)
            all_stats.update(batch_stats)
            
            for cleaned_line in cleaned_lines:
                minhash = cleaner.create_minhash(cleaned_line)
                if not lsh.query(minhash):
                    lsh.insert(f"line_{global_line_counter}", minhash)
                    fout.write(cleaned_line + '\n')
                    cleaner.stats['final_kept'] += 1
                    global_line_counter += 1
                else:
                    cleaner.stats['duplicates'] += 1
            
            for key, value in all_stats.items():
                cleaner.stats[key] = value
    
    # Finalisation
    cleaner.stats['processing_time'] = time.time() - start_time
    cleaner.print_stats()
    cleaner.save_stats()
    
    return cleaner.stats

if __name__ == "__main__":
    import sys
    
    # Permettre de choisir la configuration via argument
    config_type = sys.argv[1] if len(sys.argv) > 1 else "balanced"
    
    if config_type not in ["fast", "quality", "balanced", "custom"]:
        print("Usage: python 1clean_ultra_optimized.py [fast|quality|balanced|custom]")
        print("Par défaut: balanced")
        config_type = "balanced"
    
    main(config_type)