#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour nettoyer les fichiers de données existants en supprimant les patterns spécifiques
comme "Voici une phrase en français :" et "La phrase est :" ainsi que les guillemets superflus.
"""

import re
import os
from typing import List

# Patterns de nettoyage
generic_phrase_re = re.compile(r'^Voici une phrase en français\s*:\s*', re.IGNORECASE)
phrase_pattern_re = re.compile(r'La phrase est\s*:\s*', re.IGNORECASE)
quotes_cleanup_re = re.compile(r'^["\'“”]+|["\'“”]+$')  # Guillemets en début/fin
whitespace_re = re.compile(r'\s+')

def clean_line(line: str) -> str:
    """Nettoie une ligne en supprimant les patterns spécifiques"""
    # Suppression des patterns génériques
    line = generic_phrase_re.sub('', line)
    line = phrase_pattern_re.sub('', line)
    
    # Suppression des guillemets en début/fin
    line = quotes_cleanup_re.sub('', line)
    
    # Normalisation des espaces
    line = whitespace_re.sub(' ', line).strip()
    
    return line

def clean_file(input_path: str, output_path: str) -> dict:
    """Nettoie un fichier et retourne les statistiques"""
    stats = {
        'total_lines': 0,
        'cleaned_lines': 0,
        'empty_lines_removed': 0,
        'unchanged_lines': 0
    }
    
    print(f"Nettoyage de {input_path} vers {output_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line_num, line in enumerate(fin, 1):
            stats['total_lines'] += 1
            original_line = line.strip()
            
            if not original_line:  # Ligne vide
                continue
            
            cleaned_line = clean_line(original_line)
            
            if not cleaned_line:  # Ligne devenue vide après nettoyage
                stats['empty_lines_removed'] += 1
                continue
            
            if cleaned_line != original_line:
                stats['cleaned_lines'] += 1
            else:
                stats['unchanged_lines'] += 1
            
            fout.write(cleaned_line + '\n')
            
            # Progress
            if line_num % 10000 == 0:
                print(f"  Traité: {line_num} lignes...")
    
    return stats

def main():
    """Fonction principale"""
    # Fichiers à nettoyer
    files_to_clean = [
        ('data/all.txt', 'data/all_cleaned.txt'),
        ('data/cleaned_v4.txt', 'data/cleaned_v4_fixed.txt'),
        ('data/cleaned_v5.txt', 'data/cleaned_v5_fixed.txt')
    ]
    
    total_stats = {
        'total_lines': 0,
        'cleaned_lines': 0,
        'empty_lines_removed': 0,
        'unchanged_lines': 0
    }
    
    for input_file, output_file in files_to_clean:
        if os.path.exists(input_file):
            stats = clean_file(input_file, output_file)
            
            print(f"\nStatistiques pour {input_file}:")
            print(f"  Total lignes: {stats['total_lines']}")
            print(f"  Lignes nettoyées: {stats['cleaned_lines']}")
            print(f"  Lignes vides supprimées: {stats['empty_lines_removed']}")
            print(f"  Lignes inchangées: {stats['unchanged_lines']}")
            print(f"  Fichier de sortie: {output_file}")
            
            # Accumulation des stats
            for key in total_stats:
                total_stats[key] += stats[key]
        else:
            print(f"Fichier non trouvé: {input_file}")
    
    print(f"\n=== STATISTIQUES TOTALES ===")
    print(f"Total lignes traitées: {total_stats['total_lines']}")
    print(f"Total lignes nettoyées: {total_stats['cleaned_lines']}")
    print(f"Total lignes vides supprimées: {total_stats['empty_lines_removed']}")
    print(f"Total lignes inchangées: {total_stats['unchanged_lines']}")
    
    if total_stats['total_lines'] > 0:
        clean_ratio = (total_stats['cleaned_lines'] / total_stats['total_lines']) * 100
        print(f"Pourcentage de lignes nettoyées: {clean_ratio:.2f}%")

if __name__ == "__main__":
    main()