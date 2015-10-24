#ifndef SAMU_HPP
#define SAMU_HPP

/**
 * @brief JUDAH - Jacob is equipped with a text-based user interface
 *
 * @file samu.hpp
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
#include <string>
#include <sstream>
#include <vector>
#include <thread>
#include <condition_variable>
#include <mutex>

#include "nlp.hpp"

#include <queue>
#include <cstdio>
#include <cstring>

#include "nlp.hpp"
#include "ql.hpp"

#ifndef CHARACTER_CONSOLE
#include <pngwriter.h>
#endif
#include <chrono>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <sstream>

#ifdef DISP_CURSES
#include "disp.hpp"
#endif

class Samu
{
public:

  Samu()
  {
    cv_.notify_one();
  }

  ~Samu()
  {
    run_ = false;
    terminal_thread_.join();
  }

  bool run ( void ) const
  {
    return run_;
  }

  bool halt ( void )
  {
    run_ = false;
  }


  bool sleep ( void ) const
  {
    return sleep_;
  }

  bool sleep_after ( void ) const
  {
    return sleep_after_;
  }

  void clear_vi ( void )
  {
    vi.clear();
  }

  void FamilyCaregiverShell ( void );
  void terminal ( void )
  {
    std::unique_lock<std::mutex> lk ( mutex_ );
    cv_.wait ( lk );

    FamilyCaregiverShell();
  }

  void sentence ( int id, std::string & sentence, std::string & file )
  {
    if ( msg_mutex.try_lock() )
      {

        if ( id != old_talk_id )
          clear_vi();

        old_talk_id = id;

        SPOTriplets tv = nlp.sentence2triplets ( sentence.c_str() );

        vi << tv;

        if ( tv.size() )
          {
            std::fstream cache ( file,  std::ios_base::out|std::ios_base::app );
            if ( cache )
              {
                for ( auto t : tv )
                  cache << t << std::endl;

                cache.close();
              }
          }

        msg_mutex.unlock();

      }
    else
      {
        throw "My attention diverted elsewhere.";
      }

  }

  void sentence ( int id, std::string & sentence )
  {
    if ( msg_mutex.try_lock() )
      {

        if ( id != old_talk_id )
          clear_vi();

        old_talk_id = id;

        vi << nlp.sentence2triplets ( sentence.c_str() );

        msg_mutex.unlock();

      }
    else
      {
        throw "My attention diverted elsewhere.";
      }

  }

  void triplet ( int id, SPOTriplets & triplets )
  {
    if ( msg_mutex.try_lock() )
      {

        if ( id != old_talk_id )
          clear_vi();

        old_talk_id = id;

        vi << triplets;

        msg_mutex.unlock();

      }
    else
      {
        throw "My attention diverted elsewhere.";
      }

  }


  std::string Caregiver()
  {
    if ( caregiver_name_.size() > 0 )
      return caregiver_name_[caregiver_idx_];
    else
      return "Undefined";
  }

  void NextCaregiver()
  {
    caregiver_idx_ = ( caregiver_idx_ + 1 ) % caregiver_name_.size();
  }

  double reward ( void )
  {
    return vi.reward();
  }

  void save ( std::string & fname )
  {
#ifdef DISP_CURSES
    disp.log ( "Saving Samu..." );
#else
    std::cerr << "Saving Samu..." << std::endl;
#endif
    vi.save ( fname );
  }

  void load ( std::fstream & file )
  {
#ifdef DISP_CURSES
    disp.log ( "Loading Samu..." );
#else
    std::cerr << "Loading Samu..." << std::endl;
#endif
    vi.load ( file );
  }

  std::string get_training_file() const
  {
    return training_file;
  }

  void set_training_file ( const std::string& filename )
  {
    training_file = filename;
  }

  void set_N_e ( int N_e )
  {
    vi.set_N_e ( N_e );
  }

  void clear_N_e ( void )
  {
    vi.clearn();
  }

  void scale_N_e ( void )
  {
    vi.scalen(.65);
  }

  void scale_N_e ( double s )
  {
    vi.scalen(s);
  }
  
  int get_brel ( void )
  {
    return vi.brel();
  }

  double get_max_reward ( void ) const
  {
    return vi.get_max_reward();
  }

  double get_min_reward ( void ) const
  {
    return vi.get_min_reward();
  }

private:

  class VisualImagery
  {

  public:

    VisualImagery ( Samu & samu ) :samu ( samu )
    {}

    ~VisualImagery()
    {}

    void operator<< ( std::vector<SPOTriplet> triplets )
    {

      if ( !triplets.size() )
        return;

      for ( auto triplet : triplets )
        {
          if ( program.size() >= stmt_max )
            program.pop();

          program.push ( triplet );
        }

      if ( feelings.size() >= stmt_max )
        feelings.pop();

      feelings.push ( ql.feeling() );

      boost::posix_time::ptime now = boost::posix_time::second_clock::universal_time();

#ifndef CHARACTER_CONSOLE
      std::string image_file = "samu_vi_"+boost::posix_time::to_simple_string ( now ) +".png";
      char * image_file_p = strdup ( image_file.c_str() );
      pngwriter image ( 256, 256, 65535, image_file_p );
      free ( image_file_p );
#else
      char console[10][80];
      std::memset ( console, 0, 10*80 );
#endif

      char stmt_buffer[1024];
      char *stmt_buffer_p = stmt_buffer;

      std::queue<SPOTriplet> run = program;
      std::queue<Feeling> feels = feelings;

#ifndef Q_LOOKUP_TABLE

      std::string prg;
      stmt_counter = 0;
#ifdef PYRAMID_VI
      SPOTriplets pyramid;
#endif
      while ( !run.empty() )
        {
          auto triplet = run.front();

          prg += triplet.s.c_str();
          prg += triplet.p.c_str();
          prg += triplet.o.c_str();

#ifdef PYRAMID_VI
          pyramid.push_back ( triplet );
          SPOTriplets reverse_pyramid ( pyramid.size() );
          std::reverse_copy ( pyramid.begin(), pyramid.end(), reverse_pyramid.begin() );

          int cnt {0};
          for ( SPOTriplets::iterator it=reverse_pyramid.begin(); it!=reverse_pyramid.end() && cnt < 80; ++it )
            //while ( cnt < 80 )
            //cnt += std::snprintf ( stmt_buffer+cnt, 1024-cnt, "%s.%s(%s);", triplet.s.c_str(), triplet.p.c_str(), triplet.o.c_str() );
            cnt += std::snprintf ( stmt_buffer+cnt, 1024-cnt, "%s.%s(%s);", ( *it ).s.c_str(), ( *it ).p.c_str(), ( *it ).o.c_str() );
#elif JUSTIFY_VI
          int cnt {0};
          while ( cnt < 80 )
            cnt += std::snprintf ( stmt_buffer+cnt, 1024-cnt, "%s.%s(%s);", triplet.s.c_str(), triplet.p.c_str(), triplet.o.c_str() );
#else
          //std::snprintf ( stmt_buffer, 1024, "%s.%s(%s);", triplet.s.c_str(), triplet.p.c_str(), triplet.o.c_str() );
          if ( !feels.empty() )
            {
              auto s = feels.front();
              std::stringstream ss;
              ss << triplet.s << "." << triplet.p << "(" << triplet.o << ");";
              std::string spo = ss.str();
              std::snprintf ( stmt_buffer, 1024, "%-30s %s", spo.c_str(), s.c_str() );
            }
#endif


#ifndef CHARACTER_CONSOLE
          char font[] = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf";
          char *font_p = font;

          image.plot_text_utf8 ( font_p,
                                 11,
                                 5,
                                 256- ( ++stmt_counter ) *28,
                                 0.0,
                                 stmt_buffer_p, 0, 0, 0 );
#else

          std::strncpy ( console[stmt_counter++], stmt_buffer, 80 );

#endif

          run.pop();
          feels.pop();
        }

#ifndef CHARACTER_CONSOLE
      double *img_input = new double[256*256];

      for ( int i {0}; i<256; ++i )
        for ( int j {0}; j<256; ++j )
          {
            img_input[i*256+j] = image.dread ( i, j );
          }
#else
      double *img_input = new double[10*80];

#ifdef DISP_CURSES
      std::stringstream con;
#endif

      for ( int i {0}; i<10; ++i )
        {
#ifdef DISP_CURSES
          std::string ci;
#endif
          for ( int j {0}; j<80; ++j )
            {
              img_input[i*80+j] = ( ( double ) console[i][j] ) / 255.0;

#ifdef DISP_CURSES
              //if ( isgraph ( console[i][j] ) )
              if ( isprint ( console[i][j] ) )
                ci += console[i][j];
#endif
            }

#ifdef DISP_CURSES
          con << " " << i << ". " << ( ( ci.length() <75 ) ?ci:ci.substr ( 0, 75 ) ) << std::endl;

#endif
        }

#ifdef DISP_CURSES

#ifndef COLOR_FEELINGS
      samu.disp.vi ( con.str() );
#else      
      samu.disp.vi ( &console[0][0] );
#endif

#endif

#endif

#else
      std::string prg;
      while ( !run.empty() )
        {
          auto triplet = run.front();

          prg += triplet.s.c_str();
          prg += triplet.p.c_str();
          prg += triplet.o.c_str();

          run.pop();
        }
#endif

      auto start = std::chrono::high_resolution_clock::now();

      std::cerr << "QL start... ";

#ifndef Q_LOOKUP_TABLE

      SPOTriplet response = ql ( triplets[0], prg, img_input );

      std::stringstream resp;

      resp << samu.name
#ifdef QNN_DEBUG
           << "@"
           << ( samu.sleep_?"sleep":"awake" )
           << "."
           << ql.get_action_count()
           << "."
           << ql.get_action_relevance()
           << "%"
#endif
           <<"> "
           << response;

      std::string r = resp.str();

      std::cerr << r << std::endl;

#ifdef DISP_CURSES
      samu.disp.log ( r );
#endif

#else

      std::cerr << ql ( triplets[0], prg ) << std::endl;

#endif

      std::cerr << std::chrono::duration_cast<std::chrono::milliseconds> ( std::chrono::high_resolution_clock::now() - start ).count()
                << " ms "
                <<  std::endl;

#ifndef CHARACTER_CONSOLE

#ifndef Q_LOOKUP_TABLE
      delete[] img_input;
      image.close();
#endif

#else

#ifndef Q_LOOKUP_TABLE
      delete[] img_input;
#endif

#endif
    }

    double reward ( void )
    {
      return ql.reward();
    }

    void save ( std::string &fname )
    {
      ql.save ( fname );
    }

    void load ( std::fstream & file )
    {
      ql.load ( file );
    }

    void clear ( void )
    {
      while ( !program.empty() )
        {
          program.pop();
        }
    }

    void set_N_e ( int N_e )
    {
      ql.set_N_e ( N_e );
    }

    void clearn ( void )
    {
      ql.clearn();
    }

    void scalen ( double s )
    {
      ql.scalen(s);
    }

    int brel ( void )
    {
      return ql.get_action_relevance();
    }

    double get_max_reward ( void ) const
    {
      return ql.get_max_reward();
    }

    double get_min_reward ( void ) const
    {
      return ql.get_min_reward();
    }

  private:

    Samu &samu;
    QL ql;
    std::queue<SPOTriplet> program;
    std::queue<Feeling> feelings;
    int stmt_counter {0};
    static const int stmt_max = 10;

  };

#ifdef DISP_CURSES
  static Disp disp;
#endif
  static std::string name;

  bool run_ {true};
  bool sleep_ {true};
  int sleep_after_ {160};
  unsigned int read_usec_ {50*1000};
  std::mutex mutex_;
  std::condition_variable cv_;
  std::thread terminal_thread_ {&Samu::terminal, this};

  NLP nlp;
  VisualImagery vi {*this};

  int caregiver_idx_ {0};
  std::vector<std::string> caregiver_name_ {"Norbi", "Nandi", "Matyi", "Greta"};

  std::mutex msg_mutex;
  int old_talk_id {-std::numeric_limits<int>::max() };

  std::string training_file;
};

#endif
