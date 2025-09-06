#!/usr/bin/env python3
"""Main script for CLIP Features Retrieval System with SuperGlobal reranking"""

import argparse
import json
import os
import sys
from pathlib import Path

from retrieval_pipeline import RetrievalPipeline
from config import Config


def build_index_command(args):
    """Build FAISS index from pre-extracted CLIP features"""
    print(f"Building index from CLIP features in: {args.features_dir}")
    
    # Initialize pipeline
    pipeline = RetrievalPipeline(
        model_name=args.model,
        initial_candidates_m=args.initial_candidates
    )
    
    # Build index from pre-extracted features
    pipeline.build_index_from_clip_features(
        args.features_dir,
        index_type=args.index_type
    )
    
    # Save index
    index_path = args.index_path or Config.INDEX_PATH
    metadata_path = args.metadata_path or Config.METADATA_PATH
    
    pipeline.save_index(index_path, metadata_path)
    
    # Print stats
    stats = pipeline.get_stats()
    print("\nIndex built successfully!")
    print(f"Total embeddings: {stats['total_embeddings']}")
    print(f"Embedding dimension: {stats['embedding_dim']}")
    print(f"Index type: {stats['index_type']}")


def query_command(args):
    """Query the index with text"""
    # Load pipeline
    pipeline = RetrievalPipeline(
        model_name=args.model,
        initial_candidates_m=args.initial_candidates,
        qe_neighbors_r=args.qe_neighbors,
        refinement_neighbors_l=args.refinement_neighbors
    )
    
    # Load index
    index_path = args.index_path or Config.INDEX_PATH
    metadata_path = args.metadata_path or Config.METADATA_PATH
    
    # Check for index files (could be .bin or .npy depending on backend)
    alternative_index_path = index_path.replace('.bin', '.npy')
    if not ((os.path.exists(index_path) or os.path.exists(alternative_index_path)) and os.path.exists(metadata_path)):
        print("Index not found. Please build index first with 'build' command.")
        return
    
    pipeline.load_index(index_path, metadata_path)
    
    # Perform query
    print(f"Querying with text: '{args.query}'")
    query = args.query
    
    # Retrieve results
    scores, results = pipeline.retrieve(
        query,
        top_k=args.top_k,
        use_reranking=not args.no_reranking
    )
    
    # Display results
    rerank_status = "without" if args.no_reranking else "with"
    print(f"\nTop {len(results)} results ({rerank_status} reranking):")
    print("-" * 60)
    
    for i, (score, result) in enumerate(zip(scores, results)):
        print(f"Rank {i+1:2d}: {result.get('filename', 'Unknown')}")
        print(f"         Score: {score:.4f}")
        
        if 'initial_score' in result and not args.no_reranking:
            print(f"         Initial score: {result['initial_score']:.4f}")
        
        if 'path' in result:
            print(f"         Path: {result['path']}")
        
        print()
    
    # Save results if requested
    if args.output:
        output_data = {
            'query': args.query,
            'reranking_used': not args.no_reranking,
            'results': []
        }
        
        for i, (score, result) in enumerate(zip(scores, results)):
            result_data = result.copy()
            result_data['rank'] = i + 1
            result_data['score'] = score
            output_data['results'].append(result_data)
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to: {args.output}")


def stats_command(args):
    """Show index statistics"""
    pipeline = RetrievalPipeline()
    
    index_path = args.index_path or Config.INDEX_PATH
    metadata_path = args.metadata_path or Config.METADATA_PATH
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        print("Index not found. Please build index first with 'build' command.")
        return
    
    pipeline.load_index(index_path, metadata_path)
    stats = pipeline.get_stats()
    
    print("Index Statistics:")
    print("-" * 30)
    for key, value in stats.items():
        if key == 'reranker_config':
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="CLIP Features Retrieval System with SuperGlobal Reranking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build index from pre-extracted CLIP features
  python main.py build --features-dir clip-features-32/
  
  # Query with text
  python main.py query "person walking on beach"
  
  # Query without reranking
  python main.py query "sunset over mountains" --no-reranking
  
  # Show index statistics
  python main.py stats
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build FAISS index')
    build_parser.add_argument('--features-dir', required=True, 
                             help='Directory containing pre-extracted CLIP features (.npy files)')
    build_parser.add_argument('--model', default=Config.EMBEDDING_MODEL,
                             help='Hugging Face model name')
    build_parser.add_argument('--index-type', default=Config.INDEX_TYPE,
                             choices=['flat', 'ivf', 'hnsw'],
                             help='FAISS index type')
    build_parser.add_argument('--initial-candidates', type=int,
                             default=Config.INITIAL_CANDIDATES_M,
                             help='Number of initial candidates to retrieve')
    build_parser.add_argument('--index-path', help='Path to save index file')
    build_parser.add_argument('--metadata-path', help='Path to save metadata file')
    
    # Query command  
    query_parser = subparsers.add_parser('query', help='Query the index')
    query_parser.add_argument('query', help='Text query')
    query_parser.add_argument('--model', default=Config.EMBEDDING_MODEL,
                             help='Hugging Face model name')
    query_parser.add_argument('--top-k', type=int, default=Config.TOP_K_RESULTS,
                             help='Number of results to return')
    query_parser.add_argument('--initial-candidates', type=int,
                             default=Config.INITIAL_CANDIDATES_M,
                             help='Number of initial candidates to retrieve')
    query_parser.add_argument('--qe-neighbors', type=int,
                             default=Config.QE_NEIGHBORS_R,
                             help='Number of neighbors for query expansion')
    query_parser.add_argument('--refinement-neighbors', type=int,
                             default=Config.REFINEMENT_NEIGHBORS_L,
                             help='Number of neighbors for refinement')
    query_parser.add_argument('--no-reranking', action='store_true',
                             help='Disable SuperGlobal reranking')
    query_parser.add_argument('--index-path', help='Path to index file')
    query_parser.add_argument('--metadata-path', help='Path to metadata file')
    query_parser.add_argument('--output', help='Save results to JSON file')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show index statistics')
    stats_parser.add_argument('--index-path', help='Path to index file')
    stats_parser.add_argument('--metadata-path', help='Path to metadata file')
    
    args = parser.parse_args()
    
    if args.command == 'build':
        build_index_command(args)
    elif args.command == 'query':
        query_command(args)
    elif args.command == 'stats':
        stats_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()