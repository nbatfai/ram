#ifndef QL_HPP
#define QL_HPP

/**
 * @brief JUDAH - Jacob is equipped with a text-based user interface
 *
 * @file ql.hpp
 * @author  Norbert Bátfai <nbatfai@gmail.com>
 * @version 0.0.1
 *
 * @section LICENSE
 *
 * Copyright (C) 2015 Norbert Bátfai, batfai.norbert@inf.unideb.hu
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * @section DESCRIPTION
 *
 * JACOB, https://github.com/nbatfai/jacob
 *
 * "The son of Isaac is Jacob." The project called Jacob is an experiment
 * to replace Isaac's (GUI based) visual imagination with a character console.
 *
 * ISAAC, https://github.com/nbatfai/isaac
 *
 * "The son of Samu is Isaac." The project called Isaac is a case study
 * of using deep Q learning with neural networks for predicting the next
 * sentence of a conversation.
 *
 * SAMU, https://github.com/nbatfai/samu
 *
 * The main purpose of this project is to allow the evaluation and
 * verification of the results of the paper entitled "A disembodied
 * developmental robotic agent called Samu Bátfai". It is our hope
 * that Samu will be the ancestor of developmental robotics chatter
 * bots that will be able to chat in natural language like humans do.
 *
 */

#include <iostream>
#include <cstdarg>
#include <map>
#include <iterator>
#include <cmath>
#include <random>
#include <limits>
#include <fstream>

#include "nlp.hpp"
#include "qlc.h"

#ifndef Q_LOOKUP_TABLE
class Perceptron
{
public:
  Perceptron ( int nof, ... )
  {
    n_layers = nof;

    units = new double*[n_layers];
    n_units = new int[n_layers];

    va_list vap;

    va_start ( vap, nof );

    for ( int i {0}; i < n_layers; ++i )
      {
        n_units[i] = va_arg ( vap, int );

        if ( i )
          units[i] = new double [n_units[i]];
      }

    va_end ( vap );

    weights = new double**[n_layers-1];

#ifndef RND_DEBUG
    std::random_device init;
    std::default_random_engine gen {init() };
#else
    std::default_random_engine gen;
#endif

//    std::uniform_real_distribution<double> dist ( -1.0, 1.0 );
    std::uniform_real_distribution<double> dist ( -5.0, 5.0 );

    for ( int i {1}; i < n_layers; ++i )
      {
        weights[i-1] = new double *[n_units[i]];

        for ( int j {0}; j < n_units[i]; ++j )
          {
            weights[i-1][j] = new double [n_units[i-1]];

            for ( int k {0}; k < n_units[i-1]; ++k )
              {
                weights[i-1][j][k] = dist ( gen );
              }
          }
      }
  }

  Perceptron ( std::fstream & file )
  {
    file >> n_layers;

    units = new double*[n_layers];
    n_units = new int[n_layers];

    for ( int i {0}; i < n_layers; ++i )
      {
        file >> n_units[i];

        if ( i )
          units[i] = new double [n_units[i]];
      }

    weights = new double**[n_layers-1];

    for ( int i {1}; i < n_layers; ++i )
      {
        weights[i-1] = new double *[n_units[i]];

        for ( int j {0}; j < n_units[i]; ++j )
          {
            weights[i-1][j] = new double [n_units[i-1]];

            for ( int k {0}; k < n_units[i-1]; ++k )
              {
                file >> weights[i-1][j][k];
              }
          }
      }
  }


  double sigmoid ( double x )
  {
    return 1.0/ ( 1.0 + exp ( -x ) );
  }


  double operator() ( double image [] )
  {

    units[0] = image;

    for ( int i {1}; i < n_layers; ++i )
      {

#ifdef CUDA_PRCPS

        cuda_layer ( i, n_units, units, weights );

#else

        #pragma omp parallel for
        for ( int j = 0; j < n_units[i]; ++j )
          {
            units[i][j] = 0.0;
	    //units[i][j] = -0.5;

            for ( int k = 0; k < n_units[i-1]; ++k )
              {
                units[i][j] += weights[i-1][j][k] * units[i-1][k];
              }

            units[i][j] = sigmoid ( units[i][j] );

          }

#endif

      }

    return sigmoid ( units[n_layers - 1][0] );

  }

  void learning ( double image [], double q, double prev_q )
  {
    double y[1] {q};

    learning ( image, y );
  }

  void learning ( double image [], double y[] )
  {
    //( *this ) ( image );

    units[0] = image;

    double ** backs = new double*[n_layers-1];

    for ( int i {0}; i < n_layers-1; ++i )
      {
        backs[i] = new double [n_units[i+1]];
      }

    int i {n_layers-1};

    for ( int j {0}; j < n_units[i]; ++j )
      {
        backs[i-1][j] = sigmoid ( units[i][j] ) * ( 1.0-sigmoid ( units[i][j] ) ) * ( y[j] - units[i][j] );

        for ( int k {0}; k < n_units[i-1]; ++k )
          {
            weights[i-1][j][k] += ( 0.2* backs[i-1][j] *units[i-1][k] );
          }

      }

    for ( int i {n_layers-2}; i >0 ; --i )
      {

        #pragma omp parallel for
        for ( int j =0; j < n_units[i]; ++j )
          {

            double sum = 0.0;

            for ( int l = 0; l < n_units[i+1]; ++l )
              {
                sum += 0.19*weights[i][l][j]*backs[i][l];
              }

            backs[i-1][j] = sigmoid ( units[i][j] ) * ( 1.0-sigmoid ( units[i][j] ) ) * sum;

            for ( int k = 0; k < n_units[i-1]; ++k )
              {
                weights[i-1][j][k] += ( 0.19* backs[i-1][j] *units[i-1][k] );
              }
          }
      }

    for ( int i {0}; i < n_layers-1; ++i )
      {
        delete [] backs[i];
      }

    delete [] backs;

  }

  ~Perceptron()
  {
    for ( int i {1}; i < n_layers; ++i )
      {
        for ( int j {0}; j < n_units[i]; ++j )
          {
            delete [] weights[i-1][j];
          }

        delete [] weights[i-1];
      }

    delete [] weights;

    for ( int i {0}; i < n_layers; ++i )
      {
        if ( i )
          delete [] units[i];
      }

    delete [] units;
    delete [] n_units;

  }

  void save ( std::fstream & out )
  {
    out << " "
        << n_layers;

    for ( int i {0}; i < n_layers; ++i )
      out << " " << n_units[i];

    for ( int i {1}; i < n_layers; ++i )
      {
        for ( int j {0}; j < n_units[i]; ++j )
          {
            for ( int k {0}; k < n_units[i-1]; ++k )
              {
                out << " "
                    << weights[i-1][j][k];

              }
          }
      }

  }

private:
  Perceptron ( const Perceptron & );
  Perceptron & operator= ( const Perceptron & );

  int n_layers;
  int* n_units;
  double **units;
  double ***weights;

};
#endif

typedef std::string Feeling;

class QL
{
public:
  QL ( )
  {

    std::random_device init;
    std::default_random_engine gen {init() };
    //std::uniform_int_distribution<int> dist ( 0, 1 );
    //std::uniform_int_distribution<int> dist ( 0, 6 );
    std::uniform_int_distribution<int> dist ( 0, 9 );
    //std::uniform_int_distribution<int> dist (48, 126 );
    for ( int i {0}; i<10; ++i )
      {
        std::stringstream ss;
        for ( int j {0}; j<40; ++j )
          {
	    //char c = dist ( gen );
	    //ss << c;
            ss << dist ( gen );
          }
        prcps_f[ss.str()] = new Perceptron ( 3, 10*80, 16,  1 ); //exp.a1 // 302
      }
  }

  QL ( SPOTriplet triplet )
  {}

  ~QL()
  {
#ifndef Q_LOOKUP_TABLE
    for ( std::map<SPOTriplet, Perceptron*>::iterator it=prcps.begin(); it!=prcps.end(); ++it )
      delete it->second;
#endif
  }

  double f ( double u, int n )
  {
    if ( n < N_e )
      return max_reward;
    else
      return u;
  }

#ifdef QNN_DEBUG_BREL
  int get_action_count() const
  {
    return frqs.size();
  }

  int get_action_relevance() const
  {
    return relevance*100.0;
  }
#endif

#ifndef Q_LOOKUP_TABLE

  double max_ap_Q_sp_ap ( double image[] )
  {
    double q_spap;
    double min_q_spap = -std::numeric_limits<double>::max();

    for ( std::map<SPOTriplet, Perceptron*>::iterator it=prcps.begin(); it!=prcps.end(); ++it )
      {

        q_spap = ( * ( it->second ) ) ( image );
        if ( q_spap > min_q_spap )
          min_q_spap = q_spap;
      }

    return min_q_spap;
  }

  double max_ap_Q_sp_ap_f ( double image[] )
  {
    double q_spap;
    double min_q_spap = -std::numeric_limits<double>::max();

    for ( std::map<Feeling, Perceptron*>::iterator it=prcps_f.begin(); it!=prcps_f.end(); ++it )
      {

        q_spap = ( * ( it->second ) ) ( image );
        if ( q_spap > min_q_spap )
          min_q_spap = q_spap;
      }

    return min_q_spap;
  }

  SPOTriplet argmax_ap_f ( std::string prg, double image[] )
  {
    double min_f = -std::numeric_limits<double>::max();
    SPOTriplet ap;

#ifdef QNN_DEBUG_BREL
    double sum {0.0}, rel;
    double a = std::numeric_limits<double>::max(), b = -std::numeric_limits<double>::max();
#endif

    for ( std::map<SPOTriplet, Perceptron*>::iterator it=prcps.begin(); it!=prcps.end(); ++it )
      {

        double  q_spap = ( * ( it->second ) ) ( image );
        double explor = f ( q_spap, frqs[it->first][prg] );

#ifdef QNN_DEBUG_BREL
        sum += q_spap;

        if ( q_spap > b )
          b = q_spap;

        if ( q_spap < a )
          a = q_spap;
#endif

        if ( explor >= min_f )
          {
            min_f = explor;
            ap = it->first;
#ifdef QNN_DEBUG_BREL
            rel = q_spap;
#endif
          }
      }
#ifdef QNN_DEBUG
    relevance = ( rel - sum/ ( ( double ) prcps.size() ) ) / ( b-a );
#endif

    return ap;
  }


  Feeling argmax_ap_f_f ( std::string prg, double image[] )
  {
    double min_f = -std::numeric_limits<double>::max();
    Feeling ap;

#ifdef QNN_DEBUG_BREL
    double sum {0.0}, rel;
    double a = std::numeric_limits<double>::max(), b = -std::numeric_limits<double>::max();
#endif

    for ( std::map<Feeling, Perceptron*>::iterator it=prcps_f.begin(); it!=prcps_f.end(); ++it )
      {

        double  q_spap = ( * ( it->second ) ) ( image );
        double explor = f ( q_spap, frqs_f[it->first][prg] );

#ifdef QNN_DEBUG_BREL
        sum += q_spap;

        if ( q_spap > b )
          b = q_spap;

        if ( q_spap < a )
          a = q_spap;
#endif

        if ( explor >= min_f )
          {
            min_f = explor;
            ap = it->first;
#ifdef QNN_DEBUG_BREL
            rel = q_spap;
#endif
          }
      }
#ifdef QNN_DEBUG
    relevance_f = ( rel - sum/ ( ( double ) prcps_f.size() ) ) / ( b-a );
#endif

    return ap;
  }

  SPOTriplet operator() ( SPOTriplet triplet, std::string prg, double image[] )
  {

    // Here 'triplet' will also be used as a simplified state in further developments
    // s' = triplet
    // r' = reward

    double reward =
      //3.0 * triplet.cmp ( prev_action ) - 1.5;
      //( triplet == prev_action ) ?1.0:-2.0;
      ( triplet == prev_action ) ?max_reward:min_reward;

    Feeling feeling = prcps_f.begin()->first;

    if ( prcps.find ( triplet ) == prcps.end() )
      {

#ifndef CHARACTER_CONSOLE
        prcps[triplet] = new Perceptron ( 3, 256*256, 80, 1 );
        //prcps[triplet] = new Perceptron ( 3, 256*256, 400, 1 );
#else
        prcps[triplet] = new Perceptron ( 3, 10*80, 32,  1 ); //exp.a1 // 302

        //prcps[triplet] = new Perceptron ( 3, 10*80, 64,  1 ); //exp.a4
        //prcps[triplet] = new Perceptron ( 4, 10*80, 256, 32,  1 );
        //prcps[triplet] = new Perceptron ( 5, 10*80, 256, 128, 32, 1 ); // 355
        //prcps[triplet] = new Perceptron ( 5, 10*80, 196, 32,  32, 1 ); // 302
        //prcps[triplet] = new Perceptron ( 5, 10*80, 400, 400,  32, 1 ); // 302
#endif
      }

    SPOTriplet action = triplet;

    if ( prev_reward >  -std::numeric_limits<double>::max() )
      {
        ++frqs[prev_action][prev_state];
        ++frqs_f[prev_feeling][prev_state];

        double max_ap_q_sp_ap = max_ap_Q_sp_ap ( image );
        double max_ap_q_sp_ap_f = max_ap_Q_sp_ap_f ( image );

        double old_q_q_s_a_nn_q_s_a;

        for ( int z {0}; z<10; ++z )
          {
            double nn_q_s_a = ( *prcps[prev_action] ) ( prev_image );

            double nn_q_s_a_f = ( *prcps_f[prev_feeling] ) ( prev_image );

            double q_q_s_a = nn_q_s_a +
                             alpha ( frqs[prev_action][prev_state] ) *
                             ( reward + gamma * max_ap_q_sp_ap - nn_q_s_a );

            double q_q_s_a_f = nn_q_s_a_f +
                               alpha ( frqs_f[prev_feeling][prev_state] ) *
                               ( reward + gamma * max_ap_q_sp_ap_f - nn_q_s_a_f );

            prcps[prev_action]->learning ( prev_image, q_q_s_a, nn_q_s_a );
            prcps_f[prev_feeling]->learning ( prev_image, q_q_s_a_f, nn_q_s_a_f );

            std::cerr << "### "
                      << q_q_s_a - nn_q_s_a
                      << " "
                      << q_q_s_a
                      << " "
                      << nn_q_s_a
                      << std::endl;

            std::cerr << "###Feelings "
                      << q_q_s_a_f - nn_q_s_a_f
                      << " "
                      << q_q_s_a_f
                      << " "
                      << nn_q_s_a_f
                      << std::endl;

            if ( std::fabs ( old_q_q_s_a_nn_q_s_a - ( q_q_s_a - nn_q_s_a ) ) <= 0.0000000001 )
              break;

            old_q_q_s_a_nn_q_s_a = q_q_s_a - nn_q_s_a;

          }

        action = argmax_ap_f ( prg, image );
        feeling = argmax_ap_f_f ( prg, image );
      }

    prev_state = prg; 		// s <- s'
    prev_reward = reward;	// r <- r'
    prev_action = action;	// a <- a'
    prev_feeling = feeling;	// a <- a'

#ifndef CHARACTER_CONSOLE
    std::memcpy ( prev_image, image, 256*256*sizeof ( double ) );
#else
    std::memcpy ( prev_image, image, 10*80*sizeof ( double ) );
#endif
    return action;
  }

#else

  double max_ap_Q_sp_ap ( std::string prg )
  {
    double q_spap;
    double min_q_spap = -std::numeric_limits<double>::max();

    for ( std::map<SPOTriplet, std::map<std::string, double>>::iterator it=table_.begin(); it!=table_.end(); ++it )
      {
        q_spap = it->second[prg];
        if ( q_spap > min_q_spap )
          min_q_spap = q_spap;
      }

    return min_q_spap;
  }

  SPOTriplet argmax_ap_f ( std::string prg )
  {
    double q_spap;
    double min_f = -std::numeric_limits<double>::max();
    SPOTriplet ap;

    for ( std::map<SPOTriplet, std::map<std::string, double>>::iterator it=table_.begin(); it!=table_.end(); ++it )
      {

        q_spap = it->second[prg];

        double explor = f ( q_spap, frqs[it->first][prg] );

        if ( explor > min_f )
          {
            min_f = explor;
            ap = it->first;
          }
      }

    return ap;
  }

  SPOTriplet operator() ( SPOTriplet triplet, std::string prg )
  {

    // s' = triplet
    // r' = reward

    double reward =
      3.0*triplet.cmp ( prev_action ) - 1.5;

    SPOTriplet action = triplet;

    if ( prev_reward >  -std::numeric_limits<double>::max() )
      {

        ++frqs[prev_action][prev_state];

        table_[triplet][prg] = table_[triplet][prg];

        double max_ap_q_sp_ap = max_ap_Q_sp_ap ( prg );

        table_[prev_action][prev_state] =
          table_[prev_action][prev_state] +
          alpha ( frqs[prev_action][prev_state] ) *
          ( reward + gamma * max_ap_q_sp_ap - table_[prev_action][prev_state] );

        action = argmax_ap_f ( prg );

      }

    prev_state = prg; 		// s <- s'
    prev_reward = reward;   	// r <- r'
    prev_action = action;	// a <- a'

    return action;
  }

#endif

  double reward ( void )
  {
    return prev_reward;
  }

  Feeling feeling ( void )
  {
    return prev_feeling;
  }

  double alpha ( int n )
  {
    return 1.0/ ( ( ( double ) n ) + 1.0 );
  }

  void clearn ( void )
  {

    for ( std::map<SPOTriplet, std::map<std::string, int>>::iterator it=frqs.begin(); it!=frqs.end(); ++it )
      {

        for ( std::map<std::string, int>::iterator itt=it->second.begin(); itt!=it->second.end(); ++itt )
          {
            itt->second = 0;
          }
      }

  }

  double sigmoid ( int n )
  {
    return 1.0/ ( 1.0 + exp ( -n ) );
  }
  
  void scalen ( double s )
  {

    for ( std::map<SPOTriplet, std::map<std::string, int>>::iterator it=frqs.begin(); it!=frqs.end(); ++it )
      {

        for ( std::map<std::string, int>::iterator itt=it->second.begin(); itt!=it->second.end(); ++itt )
          {
            //itt->second -= ( itt->second / 5 );
	    itt->second *= s;
          }
      }

  }
  
  void save_prcps ( std::fstream & samuFile )
  {
    samuFile << prcps.size();

    int prev_p {0};
    for ( std::map<SPOTriplet, Perceptron*>::iterator it=prcps.begin(); it!=prcps.end(); ++it )
      {
        int p = ( std::distance ( prcps.begin(), it ) * 100 ) / prcps.size();
        if ( p > prev_p+9 )
          {
            std::cerr << "Saving Samu: "
                      << p
                      << "% (perceptrons)"
                      << std::endl;
            prev_p = p;
          }
        samuFile << " "
                 << it->first;
        it->second->save ( samuFile );
      }
  }

  void save_frqs ( std::fstream & samuFile )
  {
    samuFile << std::endl
             << frqs.size();

    int prev_p {0};
    for ( std::map<SPOTriplet, std::map<std::string, int>>::iterator it=frqs.begin(); it!=frqs.end(); ++it )
      {

        int p = ( std::distance ( frqs.begin(), it ) * 100 ) / frqs.size();
        if ( p > prev_p+9 )
          {
            std::cerr << "Saving Samu: "
                      << p
                      << "% (frequency table)"
                      << std::endl;
            prev_p = p;
          }

        samuFile << " "
                 << it->first
                 << " "
                 << it->second.size();
        for ( std::map<std::string, int>::iterator itt=it->second.begin(); itt!=it->second.end(); ++itt )
          {
            samuFile << " "
                     << itt->first
                     << " "
                     << itt->second;
          }
      }

  }

  void save ( std::string & fname )
  {
    std::fstream samuFile ( fname,  std::ios_base::out );

    save_prcps ( samuFile );
    save_frqs ( samuFile );

    samuFile.close();
  }

  void load_prcps ( std::fstream & file )
  {
    int prcpsSize {0};
    file >> prcpsSize;

    int prev_p {0};
    SPOTriplet t;
    for ( int s {0}; s< prcpsSize; ++s )
      {
        int p = ( s * 100 ) / prcpsSize;
        if ( p > prev_p+9 )
          {
            std::cerr << "Loading Samu: "
                      << p
                      << "% (perceptrons)"
                      << std::endl;
            prev_p = p;
          }


        file >> t;

        prcps[t] = new Perceptron ( file );
      }

  }

  void load_frqs ( std::fstream & file )
  {
    int frqsSize {0};
    file >> frqsSize;

    int prev_pc {0};
    int mapSize {0};
    SPOTriplet t;
    std::string p;
    int n;
    for ( int s {0}; s< frqsSize; ++s )
      {

        int pc = ( s * 100 ) / frqsSize;
        if ( pc > prev_pc+9 )
          {
            std::cerr << "Loading Samu: "
                      << pc
                      << "% (frequency table)"
                      << std::endl;
            prev_pc = pc;
          }

        file >> t;
        file >> mapSize;
        for ( int ss {0}; ss< mapSize; ++ss )
          {
            file >> p;
            file >> n;

            frqs[t][p] = n;
          }
      }
  }


  void load ( std::fstream & file )
  {
    load_prcps ( file );
    load_frqs ( file );
  }

  int get_N_e ( void ) const
  {
    return N_e;
  }

  void set_N_e ( int N_e )
  {
    this->N_e = N_e;
  }

  double get_max_reward ( void ) const
  {
    return max_reward;
  }

  double get_min_reward ( void ) const
  {
    return min_reward;
  }


private:

  int N_e = 30;

  QL ( const QL & );
  QL & operator= ( const QL & );

#ifdef Q_LOOKUP_TABLE
  double gamma = .2;
#else
  double gamma = .2;
#endif

#ifdef Q_LOOKUP_TABLE
  std::map<SPOTriplet, std::map<std::string, double>> table_;
#else
  std::map<SPOTriplet, Perceptron*> prcps;
  std::map<Feeling, Perceptron*> prcps_f;
#ifdef QNN_DEBUG
  double relevance {0.0};
  double relevance_f {0.0};
#endif
#endif

  std::map<SPOTriplet, std::map<std::string, int>> frqs;
  std::map<Feeling, std::map<std::string, int>> frqs_f;

  SPOTriplet prev_action;
  Feeling prev_feeling {"Hello, World!"};

  std::string prev_state;

  double prev_reward { -std::numeric_limits<double>::max() };
  double max_reward { 1.1 };
  double min_reward {-1.1*max_reward};

#ifndef CHARACTER_CONSOLE
  double prev_image [256*256];
#else
  double prev_image [10*80];
#endif

};

#endif
